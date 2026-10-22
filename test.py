import csv
import json
import time
from io import BytesIO
import base64
import logging
import msvcrt
from pathlib import Path

from PIL import Image
import numpy as np
from gym_donkeycar.core.sim_client import SDClient


###########################################

class SimpleClient(SDClient):

    def __init__(self, address, poll_socket_sleep_time=0.01):
        super().__init__(*address, poll_socket_sleep_time=poll_socket_sleep_time)
        self.last_image = None
        self.car_loaded = False
        self.image_index = 0
        self.dataset_dir = Path(__file__).resolve().parent / "dataset"
        self.last_dataset_capture = 0.0
        self.last_cte = None
        self.last_cte_rate = 0.0
        self.last_cte_time = None
        self.last_image_time = None
        self.car_configured = False

    @staticmethod
    def _frame_message(message):
        return message if message.endswith("\n") else f"{message}\n"

    def send(self, message):
        super().send(self._frame_message(message))

    def send_now(self, message):
        super().send_now(self._frame_message(message))

    def on_msg_recv(self, json_packet):
        if json_packet['msg_type'] == "need_car_config":
            if not self.car_configured:
                self.send_config()

        if json_packet['msg_type'] == "car_loaded":
            self.car_loaded = True

        if json_packet['msg_type'] == "telemetry":
            now = time.monotonic()
            cte = json_packet.get("cte")
            if cte is None:
                self.last_cte = None
                self.last_cte_rate = 0.0
                self.last_cte_time = None
            else:
                try:
                    cte = float(cte)
                except (TypeError, ValueError):
                    logging.warning("Ignoring invalid simulator cross-track error: %r", cte)
                    self.last_cte = None
                    self.last_cte_rate = 0.0
                    self.last_cte_time = None
                else:
                    if np.isfinite(cte):
                        if self.last_cte is not None and self.last_cte_time is not None:
                            elapsed = now - self.last_cte_time
                            if elapsed > 0:
                                self.last_cte_rate = (cte - self.last_cte) / elapsed
                        else:
                            self.last_cte_rate = 0.0
                        self.last_cte = cte
                        self.last_cte_time = now
                    else:
                        logging.warning("Ignoring non-finite simulator cross-track error: %r", cte)
                        self.last_cte = None
                        self.last_cte_rate = 0.0
                        self.last_cte_time = None

            imgString = json_packet["image"]
            image = Image.open(BytesIO(base64.b64decode(imgString)))
            self.image_index += 1
            self.last_image = np.asarray(image)
            self.last_image_time = now
            #don't have to, but to clean up the print, delete the image string.
            del json_packet["image"]

            if "imageb" in json_packet:
                imgString = json_packet["imageb"]
                image = Image.open(BytesIO(base64.b64decode(imgString)))
                np_img = np.asarray(image)
                #don't have to, but to clean up the print, delete the image string.
                del json_packet["imageb"]

            if "lidar" in json_packet:
                lidar = json_packet["lidar"]
                #don't have to, but to clean up the print, delete the lidar string.
                del json_packet["lidar"]

        if json_packet.get("msg_type") != "telemetry":
            print("got:", json_packet)

    @staticmethod
    def _lane_center(image, row_fraction):
        height, width = image.shape[:2]
        y = min(height - 1, int(height * row_fraction))
        row = image[y]

        if image.ndim == 2:
            brightness = row
            color_spread = np.zeros_like(row)
        else:
            pixels = row[:, :3].astype(np.int16)
            brightness = pixels.min(axis=1)
            color_spread = pixels.max(axis=1) - brightness

        marked = np.flatnonzero((brightness >= 145) & (color_spread <= 65))
        if marked.size == 0:
            return None

        groups = np.split(marked, np.flatnonzero(np.diff(marked) > 1) + 1)
        candidates = [
            (int(group[0]) + int(group[-1])) / 2
            for group in groups
            if 2 <= len(group) <= 14
        ]
        if not candidates:
            return None

        target_width = width * min(0.92, 0.54 + max(0.0, row_fraction - 0.52) * 2.8)
        pairs = [
            (left, right)
            for left in candidates
            for right in candidates
            if left < width * 0.48
            and right > width * 0.52
            and width * 0.30 <= right - left <= width * 0.98
        ]
        if pairs:
            left, right = min(
                pairs,
                key=lambda pair: abs((pair[1] - pair[0]) - target_width)
                + 0.2 * abs((pair[0] + pair[1]) / 2 - width / 2),
            )
            return (left + right) / 2

        left_edges = [x for x in candidates if width * 0.08 <= x < width * 0.48]
        right_edges = [x for x in candidates if width * 0.52 < x <= width * 0.92]
        if bool(left_edges) == bool(right_edges):
            return None

        if left_edges:
            return min(width, min(left_edges) + target_width / 2)
        return max(0.0, max(right_edges) - target_width / 2)

    @classmethod
    def _autopilot_controls(cls, image):
        if image is None or image.ndim not in (2, 3):
            return None

        height, width = image.shape[:2]
        yellow_centers = {}
        if image.ndim == 3 and image.shape[2] >= 3:
            pixels = image[:, :, :3].astype(np.int16)
            red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
            yellow = (
                (red >= 100)
                & (green >= 85)
                & (blue <= 120)
                & (red - blue >= 45)
                & (green - blue >= 35)
                & (np.abs(red - green) <= 75)
            )

            previous_center = width / 2
            for fraction in (0.42, 0.47, 0.52, 0.57, 0.62, 0.67, 0.72, 0.77):
                y = int(height * fraction)
                band = yellow[max(0, y - 1):min(height, y + 2), :]
                columns = np.flatnonzero(np.any(band, axis=0))
                if columns.size == 0:
                    continue

                groups = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
                candidates = [
                    (int(group[0]) + int(group[-1])) / 2
                    for group in groups
                    if 2 <= len(group) <= max(3, int(width * 0.20))
                ]
                if not candidates:
                    continue

                center = min(
                    candidates,
                    key=lambda candidate: abs(candidate - previous_center)
                    + 0.12 * abs(candidate - width / 2),
                )
                yellow_centers[fraction] = center
                previous_center = center

        if yellow_centers:
            far_centers = [
                center for fraction, center in yellow_centers.items() if fraction <= 0.57
            ]
            near_centers = [
                center for fraction, center in yellow_centers.items() if fraction >= 0.62
            ]
            detected_centers = list(yellow_centers.values())
            lane_confidence = min(1.0, len(yellow_centers) / 4.0)
        else:
            row_centers = {
                fraction: cls._lane_center(image, fraction)
                for fraction in (0.52, 0.54, 0.56, 0.58)
            }
            far_centers = [
                row_centers[y] for y in (0.52, 0.54)
                if row_centers[y] is not None
            ]
            near_centers = [
                row_centers[y] for y in (0.56, 0.58)
                if row_centers[y] is not None
            ]
            detected_centers = far_centers + near_centers
            lane_confidence = min(1.0, len(detected_centers) / 3.0)

        if not detected_centers:
            return None

        far_center = float(np.mean(far_centers or detected_centers))
        near_center = float(np.mean(near_centers or detected_centers))
        lateral_error = (near_center - width / 2) / (width / 2)
        curve_error = (near_center - far_center) / (width / 2)
        steering = float(np.clip(
            0.85 * lateral_error + 1.2 * curve_error,
            -0.75,
            0.75,
        ))
        corner_severity = min(
            1.0,
            max(abs(steering) / 0.75, 2.4 * abs(curve_error)),
        )
        speed_limit = 0.45 - 0.25 * corner_severity
        throttle = speed_limit * (0.6 + 0.4 * lane_confidence)
        return steering, throttle, corner_severity

    @staticmethod
    def _telemetry_controls(cte, cte_rate):
        cte_rate = float(np.clip(cte_rate, -3.0, 3.0))
        steering = float(np.clip(-0.35 * cte - 0.06 * cte_rate, -0.75, 0.75))
        corner_severity = min(
            1.0,
            max(abs(cte), 0.15 * abs(cte_rate)),
        )
        throttle = 0.45 - 0.25 * corner_severity
        return steering, throttle, corner_severity

    def _record_training_sample(self, steering, throttle, brake):
        now = time.monotonic()
        if self.last_image is None or now - self.last_dataset_capture < 0.2:
            return

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        image_name = f"frame_{self.image_index:06d}_{time.time_ns()}.png"
        Image.fromarray(self.last_image).save(self.dataset_dir / image_name)

        labels_path = self.dataset_dir / "labels.csv"
        write_header = not labels_path.exists() or labels_path.stat().st_size == 0
        with labels_path.open("a", newline="", encoding="utf-8") as labels_file:
            writer = csv.writer(labels_file)
            if write_header:
                writer.writerow(("image", "steering", "throttle", "brake"))
            writer.writerow((image_name, steering, throttle, brake))
        self.last_dataset_capture = now

    def send_config(self):
        '''
        send three config messages to setup car, racer, and camera
        '''
        if self.car_configured:
            return
        self.car_configured = True

        racer_name = "Reda"
        car_name = "Choupette 53"
        bio = ""
        country = "France"
        guid = "essai"

        # Racer info
        msg = {'msg_type': 'racer_info',
            'racer_name': racer_name,
            'car_name' : car_name,
            'bio' : bio,
            'country' : country,
            'guid' : guid }
        self.send_now(json.dumps(msg))
        time.sleep(0.2)

        # Car config
        # body_style : donkey | bare | car01 | cybertruck | f1
        # body_rgb  = (128, 128, 128) tuple of ints
        # car_name = "string less than 64 char"

        msg = '{ "msg_type" : "car_config", "body_style" : "car01", "body_r" : "255", "body_g" : "0", "body_b" : "255", "car_name" : "%s", "font_size" : "100" }' % (car_name)
        self.send_now(msg)

        #this sleep gives the car time to spawn. Once it's spawned, it's ready for the camera config.
        time.sleep(0.2)
        
        # Camera config
        # set any field to Zero to get the default camera setting.
        # this will position the camera right above the car, with max fisheye and wide fov
        # this also changes the img output to 255x255x1 ( actually 255x255x3 just all three channels have same value)
        # the offset_x moves camera left/right
        # the offset_y moves camera up/down
        # the offset_z moves camera forward/back
        # with fish_eye_x/y == 0.0 then you get no distortion
        # img_enc can be one of JPG|PNG|TGA
        msg = '{ "msg_type" : "cam_config", "fov" : "150", "fish_eye_x" : "0", "fish_eye_y" : "0", "img_w" : "255", "img_h" : "255", "img_d" : "3", "img_enc" : "PNG", "offset_x" : "0.0", "offset_y" : "0", "offset_z" : "0.0" }'
        self.send_now(msg)
        time.sleep(0.2)

        # Camera config B, for the second camera
        # set any field to Zero to get the default camera setting.
        # this will position the camera right above the car, with max fisheye and wide fov
        # this also changes the img output to 255x255x1 ( actually 255x255x3 just all three channels have same value)
        # the offset_x moves camera left/right
        # the offset_y moves camera up/down
        # the offset_z moves camera forward/back
        # with fish_eye_x/y == 0.0 then you get no distortion
        # img_enc can be one of JPG|PNG|TGA
        # msg = '{ "msg_type" : "cam_config_b", "fov" : "150", "fish_eye_x" : "1.0", "fish_eye_y" : "1.0", "img_w" : "255", "img_h" : "255", "img_d" : "1", "img_enc" : "JPG", "offset_x" : "3.0", "offset_y" : "3.0", "offset_z" : "0.0", "rot_x" : "90.0" }'
        msg = '{ "msg_type" : "cam_config_b", "enabled": "false"}'
        self.send_now(msg)
        time.sleep(0.2)

        # Lidar config
        # the offset_x moves camera left/right
        # the offset_y moves camera up/down
        # the offset_z moves camera forward/back
        # degPerSweepInc : as the ray sweeps around, how many degrees does it advance per sample (int)
        # degAngDown : what is the starting angle for the initial sweep compared to the forward vector
        # degAngDelta : what angle change between sweeps
        # numSweepsLevels : how many complete 360 sweeps (int)
        # maxRange : what it max distance we will register a hit
        # noise : what is the scalar on the perlin noise applied to point position
        # Here's some sample settings that similate a more sophisticated lidar:
        # msg = '{ "msg_type" : "lidar_config", "degPerSweepInc" : "2", "degAngDown" : "25", "degAngDelta" : "-1.0", "numSweepsLevels" : "25", "maxRange" : "50.0", "noise" : "0.2", "offset_x" : "0.0", "offset_y" : "1.0", "offset_z" : "1.0", "rot_x" : "0.0" }'
        # And here's some sample settings that similate a simple RpLidar A2 one level horizontal scan.
        #msg = '{ "msg_type" : "lidar_config", "degPerSweepInc" : "2", "degAngDown" : "0", "degAngDelta" : "-1.0", "numSweepsLevels" : "1", "maxRange" : "50.0", "noise" : "0.4", "offset_x" : "0.0", "offset_y" : "0.5", "offset_z" : "0.5", "rot_x" : "0.0" }'
        msg = '{ "msg_type" : "lidar_config", "enabled": "false"}'
        self.send_now(msg)
        time.sleep(0.2)


    def send_controls(self, steering, throttle, brake=0.0):
        msg = { "msg_type" : "control",
                "steering" : steering.__str__(),
                "throttle" : throttle.__str__(),
                "brake" : str(brake) }
        self.send(json.dumps(msg))

        #this sleep lets the SDClient thread poll our message and send it out.
        time.sleep(self.poll_socket_sleep_sec)

    def drive_with_keyboard(self, duration=50.0):
        """Drive manually or use image-based autopilot; manual frames are labeled for training."""
        steering = 0.0
        throttle = 0.0
        steering_step = 0.15
        steering_return_step = 0.02
        throttle_step = 0.10
        autopilot = True
        lane_lost = False
        autopilot_started = time.monotonic()
        start = time.time()
        sensor_timeout = 0.5

        print("Autopilot enabled. Press P for manual control; Q quits.")
        print("Manual driving: A/D steer, W/S throttle, X brake, Space coast.")
        print("Manual image/control samples are saved to dataset/ for training.")
        while time.time() - start < duration:
            brake = 0.0
            emergency_brake = False
            while msvcrt.kbhit():
                key = msvcrt.getwch().lower()
                if key == "q":
                    self.send_controls(0.0, 0.0)
                    return
                elif key == "p":
                    autopilot = not autopilot
                    steering = 0.0
                    throttle = 0.0
                    lane_lost = False
                    autopilot_started = time.monotonic()
                    print("Autopilot enabled." if autopilot else "Manual control enabled.")
                elif key == "x":
                    emergency_brake = True
                elif not autopilot and key == "a":
                    steering = max(-1.0, steering - steering_step)
                elif not autopilot and key == "d":
                    steering = min(1.0, steering + steering_step)
                elif not autopilot and key == "w":
                    throttle = min(1.0, throttle + throttle_step)
                elif not autopilot and key == "s":
                    throttle = max(-1.0, throttle - throttle_step)
                elif not autopilot and key == " ":
                    throttle = 0.0

            if emergency_brake:
                brake = 1.0
                throttle = 0.0
            elif autopilot:
                now = time.monotonic()
                cte_fresh = (
                    self.last_cte is not None
                    and self.last_cte_time is not None
                    and now - self.last_cte_time <= sensor_timeout
                )
                image_fresh = (
                    self.last_image is not None
                    and self.last_image_time is not None
                    and now - self.last_image_time <= sensor_timeout
                )
                visual_controls = (
                    self._autopilot_controls(self.last_image)
                    if image_fresh else None
                )

                if cte_fresh:
                    target_steering, throttle, corner_severity = self._telemetry_controls(
                        self.last_cte,
                        self.last_cte_rate,
                    )
                    if visual_controls is not None:
                        visual_steering, visual_throttle, visual_corner_severity = visual_controls
                        target_steering = (
                            0.7 * target_steering + 0.3 * visual_steering
                        )
                        throttle = min(throttle, visual_throttle)
                        corner_severity = max(corner_severity, visual_corner_severity)
                    lane_lost = False
                else:
                    if visual_controls is None:
                        target_steering = 0.0
                        throttle = 0.0
                        corner_severity = 0.0
                        brake = 1.0
                        if not lane_lost:
                            print("No fresh steering data; braking until sensor data returns.")
                        lane_lost = True
                    else:
                        target_steering, throttle, corner_severity = visual_controls
                        lane_lost = False

                if not lane_lost:
                    brake = max(
                        brake,
                        0.45 * float(np.clip(
                            (corner_severity - 0.55) / 0.45,
                            0.0,
                            1.0,
                        )),
                    )
                    throttle *= 1.0 - brake

                launch_progress = min(
                    1.0,
                    (time.monotonic() - autopilot_started) / 4.0,
                )
                throttle *= 0.2 + 0.8 * launch_progress
                steering_limit = 0.30 + 0.45 * min(1.0, launch_progress * 2.0)
                target_steering = float(np.clip(
                    target_steering,
                    -steering_limit,
                    steering_limit,
                ))
                if abs(target_steering) < 0.05:
                    target_steering = 0.0
                steering_change = float(np.clip(
                    target_steering - steering,
                    -(0.04 + 0.05 * corner_severity),
                    0.04 + 0.05 * corner_severity,
                ))
                steering += steering_change

            self.send_controls(steering, throttle, brake)
            if not autopilot:
                self._record_training_sample(steering, throttle, brake)
                # Let go of A/D and steering gradually returns to center.
                if steering > 0.0:
                    steering = max(0.0, steering - steering_return_step)
                elif steering < 0.0:
                    steering = min(0.0, steering + steering_return_step)
            time.sleep(0.05)



###########################################
## Make some clients and have them connect with the simulator

def test_clients():
    logging.basicConfig(level=logging.DEBUG)

    # test params
    host = "127.0.0.1" # "trainmydonkey.com" for virtual racing server
    port = 9091
    num_clients = 1
    clients = []
    time_to_drive = 50.0


    try:
        for _ in range(num_clients):
            clients.append(SimpleClient(address=(host, port)))

        time.sleep(1)

        # Only one client needs to request the scene.
        msg = '{ "msg_type" : "load_scene", "scene_name" : "roboracingleague_1" }'
        clients[0].send_now(msg)

        # The simulator requests car configuration asynchronously. Wait until
        # its response confirms the car has spawned before sending controls.
        print("Waiting for the simulator to load the car...")
        load_deadline = time.monotonic() + 30.0
        while not clients[0].car_loaded and time.monotonic() < load_deadline:
            time.sleep(0.1)
        if not clients[0].car_loaded:
            raise TimeoutError(
                "The simulator did not report car_loaded within 30 seconds. "
                "Check that the simulator is open and has finished loading."
            )

        print("Car loaded. Starting autopilot.")
        clients[0].drive_with_keyboard(duration=time_to_drive)
        time.sleep(1.0)
    finally:
        print("Waiting for message loops to stop.")
        for client in clients:
            client.stop()
        print("Clients stopped.")



if __name__ == "__main__":
    test_clients()
