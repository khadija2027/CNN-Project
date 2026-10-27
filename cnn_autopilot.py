import base64
import io
import json
import logging
import time

import numpy as np
import torch
from PIL import Image
from torch import nn

from Data_builder import SimpleClient


###########################################
## The network: must match the DrivingCNN trained in cnn.ipynb

CROP_TOP = 40             # rows of ceiling removed, as in cnn.ipynb
IMAGE_SIZE = (160, 120)   # (width, height) of the frames the model was trained on


class DrivingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1),
                                 nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 64))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(64 * 5 * 10, 64), nn.ReLU(),
                                  nn.Linear(64, 2))

    def forward(self, x):
        out = self.head(self.features(x))
        return torch.stack([torch.tanh(out[:, 0]), torch.sigmoid(out[:, 1])], dim=1)


def preprocess(image_b64):
    """Simulator frame (base64 PNG) -> 1x3x80x160 tensor, exactly as in training."""
    image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    if image.size != IMAGE_SIZE:
        image = image.resize(IMAGE_SIZE)
    array = np.asarray(image)[CROP_TOP:]
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0


###########################################

class CNNClient(SimpleClient):
    """Drive from the camera image only: the CNN chooses steering and throttle.

    The track map is NOT used for driving. The simulator's cte and speed are
    only used as safety nets: a speed cap, and a reset when the car leaves the
    track or gets stuck.
    """

    MODEL_FILE = "driving_cnn.pt"
    MAX_THROTTLE = 1.0        # scale down (e.g. 0.6) for a careful first test
    MAX_SPEED = 25.0          # brake above this measured speed
    STEER_SMOOTHING = 0.0     # 0 = raw CNN output, closer to 1 = smoother steering
    IMAGE_TIMEOUT = 0.5       # seconds without a new frame before braking

    def load_model(self):
        self.model = DrivingCNN()
        self.model.load_state_dict(torch.load(self.MODEL_FILE, map_location="cpu"))
        self.model.eval()
        logging.info("Loaded %s.", self.MODEL_FILE)

    def drive_cnn(self, duration=120.0):
        self.load_model()
        print("Resetting car to the track start...")
        self.send_now(json.dumps({"msg_type": "reset_car"}))
        time.sleep(1.0)

        steering = 0.0
        start = time.monotonic()
        last_seq = self.image_seq
        last_image_time = start
        last_progress_pos, last_progress_time = None, start
        off_track_since = None
        resets = 0
        inference_ms = 0.0
        next_status_log = start

        print(f"CNN autopilot driving for {duration:g} seconds.")
        while time.monotonic() - start < duration:
            self._check_connection()
            now = time.monotonic()

            # --- Only act on a new camera frame; no frame for too long -> stop
            if self.image_seq == last_seq or self.last_image is None:
                if now - last_image_time > self.IMAGE_TIMEOUT:
                    self.send_controls(0.0, 0.0, 1.0)
                time.sleep(0.005)
                continue
            last_seq = self.image_seq
            last_image_time = now

            # --- CNN: image -> steering, throttle
            t0 = time.perf_counter()
            with torch.no_grad():
                cnn_steering, cnn_throttle = self.model(preprocess(self.last_image))[0].tolist()
            inference_ms = (time.perf_counter() - t0) * 1000.0

            steering = (self.STEER_SMOOTHING * steering
                        + (1.0 - self.STEER_SMOOTHING) * cnn_steering)
            throttle = min(cnn_throttle, self.MAX_THROTTLE)
            brake = 0.0

            # --- Safety net: speed cap from the measured speed
            speed = self.last_speed if self.last_speed is not None else 0.0
            if speed > self.MAX_SPEED:
                throttle, brake = 0.0, 0.3

            # --- Recovery: off the track or stuck -> reset to the start
            try:
                cte = abs(float(self.last_cte))
            except (TypeError, ValueError):
                cte = 0.0
            if cte > self.OFF_TRACK_DISTANCE:
                off_track_since = off_track_since or now
            else:
                off_track_since = None
            if self.position is not None:
                pos = np.array(self.position)
                if last_progress_pos is None or np.hypot(*(pos - last_progress_pos)) > 2.0:
                    last_progress_pos, last_progress_time = pos, now
            stuck = (now - last_progress_time > self.STUCK_SECONDS
                     and abs(speed) < self.STUCK_MIN_SPEED)
            off = off_track_since is not None and now - off_track_since > 1.5
            if stuck or off:
                resets += 1
                logging.warning("Car %s after %.1fs; resetting to track start (reset #%d).",
                                "stuck" if stuck else "off track", now - start, resets)
                self.send_now(json.dumps({"msg_type": "reset_car"}))
                time.sleep(1.0)
                steering = 0.0
                last_progress_pos, last_progress_time = None, time.monotonic()
                off_track_since = None
                last_seq = self.image_seq
                last_image_time = time.monotonic()
                continue

            self.send_controls(steering, throttle, brake)

            if now >= next_status_log:
                logging.info("CNN: steer=%.2f throttle=%.2f brake=%.1f speed=%.2f cte=%.2f "
                             "inference=%.1fms resets=%d",
                             steering, throttle, brake, speed, cte, inference_ms, resets)
                next_status_log = now + 3.0

        self.send_controls(0.0, 0.0, 1.0)
        logging.info("Run finished: %.0fs driven, %d resets.", duration, resets)


###########################################
## Connect, load the scene, drive

def wait_for_car(host, port, timeout=30.0):
    """Connect and wait until the car is loaded in the forced scene (as in Data_builder.py)."""
    client = CNNClient(address=(host, port))
    print(f"Waiting for the simulator to load {client.scene_name}...")
    deadline = time.monotonic() + timeout
    exit_sent = False
    while (not (client.car_loaded and client.scene_load_requested)
           and time.monotonic() < deadline):
        if client.car_loaded and not client.scene_load_requested and not exit_sent:
            # Already inside another scene: leave it and reconnect to get the menu.
            logging.info("Simulator is already in a scene; leaving it to load %s.",
                         client.scene_name)
            client.send_now(json.dumps({"msg_type": "exit_scene"}))
            time.sleep(1.5)
            client.stop()
            time.sleep(0.5)
            client = CNNClient(address=(host, port))
            exit_sent = True
        if (client.scene_selection_time is not None
                and not client.scene_names_received
                and not client.scene_load_requested
                and time.monotonic() - client.scene_selection_time >= 1.0):
            client.scene_load_requested = True
            client.scene_load_requested_at = time.monotonic()
            client.send_now(json.dumps({"msg_type": "load_scene",
                                        "scene_name": client.scene_name}))
        if (client.scene_load_requested
                and client.scene_load_requested_at is not None
                and not client.car_configured
                and time.monotonic() - client.scene_load_requested_at >= 2.0):
            client.send_config()
        time.sleep(0.1)
    if not client.car_loaded:
        client.stop()
        raise TimeoutError(
            f"The simulator did not report car_loaded within {timeout:g} seconds. "
            "Check that the simulator is open and has finished loading."
        )
    return client


def main():
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(2)   # leave CPU for the simulator
    client = None
    try:
        client = wait_for_car("127.0.0.1", 9091)
        print("Car loaded. Starting CNN autopilot.")
        client.drive_cnn(duration=120.0)
        time.sleep(1.0)
    finally:
        if client is not None:
            client.stop()
        print("Client stopped.")


if __name__ == "__main__":
    main()
