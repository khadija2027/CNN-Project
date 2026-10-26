import base64
import csv
import json
import os
import queue
import time
import logging
from collections import deque
from threading import Lock, Thread

import numpy as np
from gym_donkeycar.core.sim_client import SDClient


###########################################

class DatasetRecorder:
    """Write camera frames + driving labels to disk on a background thread.

    Layout:  <root>/<session>/images/000000.png  and  <root>/<session>/labels.csv
    """

    FIELDS = ["image", "steering", "throttle", "brake", "speed", "target_speed",
              "off_center", "lap", "speed_scale", "timestamp"]

    def __init__(self, root="dataset"):
        session = time.strftime("%Y%m%d_%H%M%S")
        self.directory = os.path.join(root, session)
        self.image_dir = os.path.join(self.directory, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        self.count = 0
        self._queue = queue.Queue()
        self._csv_file = open(os.path.join(self.directory, "labels.csv"), "w", newline="")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=self.FIELDS)
        self._csv.writeheader()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()
        logging.info("Recording dataset to %s", self.directory)

    def write(self, image_b64, labels):
        name = f"{self.count:06d}.png"
        self.count += 1
        self._queue.put((name, image_b64, labels))

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            name, image_b64, labels = item
            with open(os.path.join(self.image_dir, name), "wb") as f:
                f.write(base64.b64decode(image_b64))
            self._csv.writerow({"image": f"images/{name}", **labels})
            if self.count % 50 == 0:
                self._csv_file.flush()

    def close(self):
        self._queue.put(None)
        self._thread.join()
        self._csv_file.close()
        logging.info("Dataset closed: %d frames in %s", self.count, self.directory)


###########################################

class ObstacleMap:
    """Static obstacles in world coordinates (x, z), built from lidar hits.

    Occupancy grid: each hit increments the count of its cell, and a cell is an
    obstacle once it has been hit MIN_HITS times (filters single noisy returns).
    Fixed cells keep the real outline of objects: two cones side by side stay
    two obstacles instead of being averaged into a phantom one in the gap.
    """

    def __init__(self, cell_size=0.5, min_hits=3, max_cells=5000):
        self.cell_size = cell_size
        self.min_hits = min_hits
        self.max_cells = max_cells
        self.counts = {}
        self._confirmed = np.empty((0, 2))

    def add(self, point, hits=1):
        cell = (int(np.floor(point[0] / self.cell_size)),
                int(np.floor(point[1] / self.cell_size)))
        count = self.counts.get(cell)
        if count is None and len(self.counts) >= self.max_cells:
            return False
        count = (count or 0) + hits
        self.counts[cell] = count
        newly_confirmed = count >= self.min_hits and count - hits < self.min_hits
        if newly_confirmed:
            center = (np.array(cell) + 0.5) * self.cell_size
            self._confirmed = np.vstack([self._confirmed, center])
        return newly_confirmed

    def confirmed(self):
        return self._confirmed


###########################################

class SimpleClient(SDClient):

    # All distances/speeds below are in TRACK units: the units of the centerline
    # nodes, of the simulator's cte and of the lidar. Some simulator builds
    # report pos_x/pos_z/speed divided by 8; that is detected and undone in
    # _calibrate_position_scale, so the whole controller works in one unit.
    POSITION_SCALE_CANDIDATES = (1.0, 8.0)

    # ---- Tuning constants -------------------------------------------------
    STEER_SIGN = 1.0          # set to -1.0 if the car steers AWAY from the path
    STEER_GAIN = 1.2          # lower gain reduces oscillation and tight circles
    LOOKAHEAD_MIN = 5.0
    LOOKAHEAD_MAX = 16.0
    LOOKAHEAD_SPEED_GAIN = 0.6  # seconds of travel added to the lookahead
    LOOKAHEAD_CORNER_FACTOR = 0.8  # max lookahead past a corner entry, times its radius
    # Measured in the simulator: full throttle ~17/s^2 up to 24+, braking
    # 13/s^2 at brake 0.5, 16/s^2 at 0.7, 24/s^2 at 1.0 (all stable).
    MAX_SPEED = 25.0          # straight-line cap; NOT scaled: braking distance protects it
    MAX_LATERAL_ACCEL = 12.0  # corner speed limit at speed_scale = 1
    MAX_BRAKING_ACCEL = 12.0  # planned deceleration before corners (half of measured max)
    MAX_THROTTLE = 1.0
    THROTTLE_KP = 0.05        # throttle per unit of speed error
    THROTTLE_KI = 0.02        # integral gain: holds higher speeds against drag
    BRAKE_KP = 0.25           # brake per unit of overspeed
    BRAKE_LATENCY = 0.2       # seconds between a command and the car reacting
    MAX_STEER_RATE = 0.12     # max steering change per control step (smoothing)
    OFF_TRACK_DISTANCE = 9.0  # distance from centerline before we consider the car lost
    STUCK_SECONDS = 4.0
    STUCK_MIN_SPEED = 1.0     # only call it stuck when the car is actually stopped

    # ---- Adaptive speed search (one adjustment per lap) -------------------
    # speed_scale multiplies (squared) MAX_LATERAL_ACCEL, so every corner speed
    # scales linearly with it. Straight-line speed is limited by MAX_SPEED and
    # the braking distance to the next corner only.
    SPEED_UP_FACTOR = 1.10    # after a clean lap
    CRASH_FACTOR = 0.85       # after leaving the track / getting stuck
    WIDE_LAP_FACTOR = 0.97    # after a lap that came too close to the edge
    SAFE_OFFSET = 3.5         # max error from the planned path for a lap to count as clean
    CEILING_MARGIN = 0.95     # stay this far below the speed that crashed
    CEILING_RETRY_LAPS = 3    # clean laps at the ceiling before trying 5% above it
    MIN_SPEED_SCALE = 0.4     # never slow down below this, whatever happens
    DISCARD_BEFORE_CRASH = 2.0  # seconds of frames dropped before a reset
    TUNING_FILE = "autopilot_tuning.json"

    # ---- Obstacle avoidance (lidar hits mapped into world coordinates) ----
    # The simulator draws every lidar ray as a red line around the car and has
    # no option to hide them, so the lidar is off by default. Without it the
    # car still follows the track from its coordinates, but does not see
    # obstacles. Set to True on tracks with cones / obstacles.
    USE_LIDAR = False
    LIDAR_ANGLE_SIGN = 1.0      # set to -1.0 if obstacles are mapped mirrored (see logs)
    LIDAR_RANGE = 30.0
    LIDAR_HEIGHT = 0.5          # low enough to hit cones, high enough to miss the floor
    LIDAR_FORWARD_OFFSET = 0.5  # lidar position in front of the car origin
    TRACK_HALF_WIDTH = 7.0      # hits further than this from the centerline are scenery
    MAX_PATH_OFFSET = 4.0       # how far the planned path may leave the centerline
    OBSTACLE_CLEARANCE = 1.8    # car half-width + safety margin
    OBSTACLE_CELL_SIZE = 0.5    # resolution of the obstacle grid
    AVOID_HORIZON = 30.0        # road ahead considered when planning
    AVOID_BEHIND = 3.0          # keep the offset until the car body has passed
    OFFSET_RATE = 0.1           # max change of the planned offset per step
    AVOID_MAX_SPEED = 10.0      # speed cap while passing close to an obstacle
    NO_GAP_SPEED = 3.0          # crawl speed when no lane is fully clear
    CAR_HALF_WIDTH = 1.2        # corridor checked for emergency braking

    def __init__(self, address, poll_socket_sleep_time=0.01):
        super().__init__(*address, poll_socket_sleep_time=poll_socket_sleep_time)
        self.last_image = None
        self.image_seq = 0
        self.lidar_scan = None      # (position, yaw, [{"d", "rx", "ry"}, ...]) from one packet
        self.lidar_seq = 0
        self.last_hit = "none"
        self.collisions = []        # (object name, car position, yaw) per new collision
        self.last_cte = None
        self.position_scale = 1.0   # telemetry -> track units, see _calibrate_position_scale
        self.speed_scale = 1.0
        self.scale_ceiling = float("inf")
        self.clean_laps_at_ceiling = 0
        self.car_loaded = False
        self.last_speed = None
        self.position = None
        self.yaw = None
        self.total_nodes = None
        self.track_nodes = {}
        self.pending_node = None
        self.pending_node_sent_at = 0.0
        self.last_position_time = None
        self.last_sim_throttle = None
        self.last_sim_steering = None
        self.last_telemetry_fields = set()
        self.scene_name = "circuit_launch"   # forced scene (see test_clients)
        self.scene_selection_time = None
        self.scene_names_received = False
        self.scene_load_requested = False
        self.scene_load_requested_at = None
        self.car_configured = False
        self._config_lock = Lock()

    # ------------------------------------------------------------------
    def on_msg_recv(self, json_packet):
        msg_type = json_packet.get("msg_type")

        if msg_type == "scene_selection_ready":
            logging.info("Simulator is ready for scene selection.")
            self.car_loaded = False
            self.car_configured = False
            self.track_nodes = {}
            self.pending_node = None
            self.scene_selection_time = time.monotonic()
            self.send(json.dumps({"msg_type": "get_scene_names"}))

        if msg_type == "scene_names":
            self.scene_names_received = True
            scene_names = json_packet.get("scene_names", [])
            if self.scene_name not in scene_names:
                logging.error("Scene %r is not available. Simulator scenes: %s",
                              self.scene_name, scene_names)
            else:
                logging.info("Loading simulator scene %s.", self.scene_name)
                self.scene_load_requested = True
                self.scene_load_requested_at = time.monotonic()
                self.send_now(json.dumps({"msg_type": "load_scene",
                                          "scene_name": self.scene_name}))

        if msg_type == "scene_loaded":
            logging.info("Simulator scene loaded; sending car and camera configuration.")
            self.send_config()

        if msg_type == "need_car_config":
            logging.info("Simulator requested car configuration.")
            self.send_config()
            self.car_loaded = True
            logging.info("Car configuration sent; simulator is ready to drive.")

        if msg_type == "car_loaded":
            self.send_config()
            self.car_loaded = True
            logging.info("Simulator confirmed the car is loaded.")

        if msg_type == "telemetry":
            now = time.monotonic()
            self.last_telemetry_fields = set(json_packet)
            try:
                self.position = tuple(float(json_packet[k]) * self.position_scale
                                      for k in ("pos_x", "pos_z"))
                self.yaw = float(json_packet["yaw"])
                if not all(np.isfinite(v) for v in (*self.position, self.yaw)):
                    raise ValueError("non-finite vehicle coordinates")
                self.last_position_time = now
            except (KeyError, TypeError, ValueError):
                self.position = None
                self.yaw = None
                self.last_position_time = None
            try:
                if json_packet.get("totalNodes") is not None:
                    total = int(json_packet["totalNodes"])
                    if total != self.total_nodes:
                        self.track_nodes = {}   # track changed: refetch
                        self.pending_node = None
                    self.total_nodes = total
            except (TypeError, ValueError):
                logging.warning("Invalid track node count: %r", json_packet.get("totalNodes"))
            try:
                self.last_speed = float(json_packet.get("speed")) * self.position_scale
            except (TypeError, ValueError):
                self.last_speed = None
            self.last_sim_throttle = json_packet.get("throttle")
            self.last_sim_steering = json_packet.get("steering_angle")
            if json_packet.get("image"):
                self.last_image = json_packet["image"]
                self.image_seq += 1
            if self.position is not None and isinstance(json_packet.get("lidar"), list):
                self.lidar_scan = (self.position, self.yaw, json_packet["lidar"])
                self.lidar_seq += 1
            hit = json_packet.get("hit", "none")
            if hit != "none" and hit != self.last_hit and self.position is not None:
                self.collisions.append((hit, self.position, self.yaw))
            self.last_hit = hit
            self.last_cte = json_packet.get("cte")

        if msg_type == "node_position":
            try:
                index = self.pending_node
                x, z = float(json_packet["pos_x"]), float(json_packet["pos_z"])
                if index is not None and np.isfinite(x) and np.isfinite(z):
                    self.track_nodes[index] = (x, z)
                self.pending_node = None
            except (KeyError, TypeError, ValueError):
                logging.warning("Invalid track node received: %s", json_packet)
                self.pending_node = None

        if msg_type != "telemetry":
            print("got:", json_packet)

    # ------------------------------------------------------------------
    def send_config(self):
        '''
        send config messages to setup car, racer, and camera
        '''
        with self._config_lock:
            if self.car_configured:
                return

            racer_name = "Reda"
            car_name = "Choupette 53"
            body_rgb = (255, 0, 0)
            bio = ""
            country = "France"
            guid = "essai"

            msg = {'msg_type': 'racer_info',
                   'racer_name': racer_name,
                   'car_name': car_name,
                   'bio': bio,
                   'country': country,
                   'guid': guid}
            self.send_now(json.dumps(msg))
            logging.info("Sent racer_info configuration.")
            time.sleep(0.2)

            msg = {
                "msg_type": "car_config",
                "body_style": "car01",
                "body_r": str(body_rgb[0]),
                "body_g": str(body_rgb[1]),
                "body_b": str(body_rgb[2]),
                "car_name": car_name,
                "font_size": "100",
            }
            self.send_now(json.dumps(msg))
            logging.info("Sent car_config for %s.", car_name)
            time.sleep(0.2)

            msg = {
                "msg_type": "cam_config",
                "fov": "150",
                "fish_eye_x": "0",
                "fish_eye_y": "0",
                "img_w": "255",
                "img_h": "255",
                "img_d": "3",
                "img_enc": "PNG",
                "offset_x": "0.0",
                "offset_y": "0",
                "offset_z": "0.0",
            }
            self.send_now(json.dumps(msg))
            logging.info("Sent cam_config.")
            time.sleep(0.2)

            self.send_now(json.dumps({"msg_type": "cam_config_b", "enabled": "false"}))
            time.sleep(0.2)
            if not self.USE_LIDAR:
                self.send_now(json.dumps({"msg_type": "lidar_config", "enabled": "false"}))
                logging.info("Lidar disabled (USE_LIDAR = False): no red rays, no obstacle map.")
                time.sleep(0.2)
                self.car_configured = True
                return
            # One horizontal 360° sweep, 2° steps, no noise: used to map obstacles.
            self.send_now(json.dumps({
                "msg_type": "lidar_config",
                "degPerSweepInc": "2.0",
                "degAngDown": "0.0",
                "degAngDelta": "-1.0",
                "numSweepsLevels": "1",
                "maxRange": str(self.LIDAR_RANGE),
                "noise": "0.0",
                "offset_x": "0.0",
                "offset_y": str(self.LIDAR_HEIGHT),
                "offset_z": str(self.LIDAR_FORWARD_OFFSET),
                "rot_x": "0.0",
            }))
            logging.info("Sent lidar_config.")
            time.sleep(0.2)
            self.car_configured = True

    # ------------------------------------------------------------------
    def send_controls(self, steering, throttle, brake=0.0):
        if self.aborted or not self.do_process_msgs:
            raise ConnectionError(
                f"Simulator connection to {self.host}:{self.port} was lost; "
                "cannot send driving controls."
            )
        msg = {"msg_type": "control",
               "steering": str(float(steering)),
               "throttle": str(float(throttle)),
               "brake": str(float(brake))}
        self.send(json.dumps(msg))
        time.sleep(self.poll_socket_sleep_sec)

    # ------------------------------------------------------------------
    def _check_connection(self):
        if self.aborted or not self.do_process_msgs:
            raise ConnectionError(
                f"Simulator connection to {self.host}:{self.port} was lost. "
                "Check the simulator logs and restart it before reconnecting."
            )

    def _request_next_node(self, last_request_time):
        """Request track nodes one at a time, retrying if a reply is lost."""
        now = time.monotonic()
        if not self.total_nodes or len(self.track_nodes) >= self.total_nodes:
            return last_request_time
        if self.pending_node is not None:
            if now - self.pending_node_sent_at < 0.5:
                return last_request_time
            self.pending_node = None          # reply lost -> retry
        if now - last_request_time < 0.03:
            return last_request_time
        for i in range(self.total_nodes):
            if i not in self.track_nodes:
                self.pending_node = i
                self.pending_node_sent_at = now
                self.send_now(json.dumps({"msg_type": "node_position", "index": str(i)}))
                return now
        return last_request_time

    def _wait_for_route(self, timeout=40.0):
        """Hold the car still until every centerline node is known."""
        logging.info("Downloading track centerline...")
        deadline = time.monotonic() + timeout
        last_req = 0.0
        while time.monotonic() < deadline:
            self._check_connection()
            last_req = self._request_next_node(last_req)
            if self.total_nodes and len(self.track_nodes) >= self.total_nodes:
                logging.info("Received %d track nodes.", self.total_nodes)
                return
            self.send_controls(0.0, 0.0, 1.0)
            time.sleep(0.02)
        raise TimeoutError(
            f"Only got {len(self.track_nodes)}/{self.total_nodes} track nodes. "
            f"Telemetry fields seen: {sorted(self.last_telemetry_fields)}"
        )

    def _project_to_route(self, points, position, start_idx=None):
        """Project the car onto a nearby centerline segment."""
        n = len(points)
        if start_idx is None:
            candidates = range(n)
        else:
            # Small window: a car moves well under one node per control step,
            # and a wide window lets the estimate jump to another part of the
            # track that passes close by (e.g. node 61 -> 86 on circuit_launch).
            candidates = ((start_idx + k) % n for k in range(-3, 9))

        best = None
        for idx in candidates:
            start, end = points[idx], points[(idx + 1) % n]
            segment = end - start
            length_sq = float(np.dot(segment, segment))
            if length_sq <= 1e-12:
                fraction = 0.0
                projected = start
            else:
                fraction = float(np.clip(np.dot(position - start, segment) / length_sq,
                                         0.0, 1.0))
                projected = start + fraction * segment
            distance = float(np.hypot(*(position - projected)))
            if best is None or distance < best[0]:
                best = (distance, idx, fraction, projected)
        return best

    def _point_ahead(self, points, start_idx, fraction, distance, segment_lengths):
        """Walk a distance along the cyclic centerline, interpolating between nodes."""
        n = len(points)
        idx = start_idx
        remaining = distance
        available = segment_lengths[idx] * (1.0 - fraction)
        if remaining <= available:
            portion = fraction + remaining / segment_lengths[idx]
            return points[idx] + portion * (points[(idx + 1) % n] - points[idx])
        remaining -= available
        idx = (idx + 1) % n
        for _ in range(n + 1):
            length = segment_lengths[idx]
            if remaining <= length:
                portion = remaining / length if length > 0 else 0.0
                return points[idx] + portion * (points[(idx + 1) % n] - points[idx])
            remaining -= length
            idx = (idx + 1) % n
        return points[idx]

    @staticmethod
    def _segment_curvature(points):
        """Unsigned curvature of each segment (the tighter of its two end nodes)."""
        n = len(points)
        node_curvature = np.zeros(n, dtype=float)
        for i in range(n):
            previous = points[i] - points[(i - 1) % n]
            following = points[(i + 1) % n] - points[i]
            previous_length = float(np.hypot(*previous))
            following_length = float(np.hypot(*following))
            chord_length = float(np.hypot(*(points[(i + 1) % n] - points[(i - 1) % n])))
            denominator = previous_length * following_length * chord_length
            if denominator > 1e-9:
                cross = previous[0] * following[1] - previous[1] * following[0]
                node_curvature[i] = 2.0 * cross / denominator

        return np.maximum(np.abs(node_curvature), np.roll(np.abs(node_curvature), -1))

    def _segment_speed_limits(self, points, segment_lengths):
        """Limit speed by the tighter curvature at either end of each segment."""
        segment_curvature = self._segment_curvature(points)
        curvature_floor = 1e-6
        return np.minimum(
            self._max_speed(),
            np.sqrt(self._max_lateral_accel() / np.maximum(segment_curvature, curvature_floor)),
        )

    def _max_speed(self):
        return self.MAX_SPEED

    def _max_lateral_accel(self):
        return self.MAX_LATERAL_ACCEL * self.speed_scale ** 2

    # ------------------------------------------------------------------
    def _load_tuning(self):
        try:
            with open(self.TUNING_FILE) as f:
                saved = json.load(f).get(self.scene_name, {})
            self.speed_scale = float(saved.get("speed_scale", self.speed_scale))
            if saved.get("scale_ceiling") is not None:
                self.scale_ceiling = float(saved["scale_ceiling"])
            logging.info("Loaded speed_scale=%.2f (ceiling %.2f) from %s.",
                         self.speed_scale, self.scale_ceiling, self.TUNING_FILE)
        except (OSError, ValueError, TypeError, AttributeError):
            pass

    def _save_tuning(self):
        try:
            with open(self.TUNING_FILE) as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        data[self.scene_name] = {"speed_scale": self.speed_scale,
                                 "scale_ceiling": (self.scale_ceiling
                                                   if np.isfinite(self.scale_ceiling) else None)}
        with open(self.TUNING_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def _adapt_after_lap(self, lap_max_offset):
        """Raise the speed after a clean lap, back off after a wide one."""
        if lap_max_offset <= self.SAFE_OFFSET:
            limit = self.scale_ceiling * self.CEILING_MARGIN
            if self.speed_scale >= limit * 0.999:
                # Converged at the ceiling: after a few clean laps, the crash
                # that set it may have been bad luck, so allow 5% more.
                self.clean_laps_at_ceiling += 1
                if self.clean_laps_at_ceiling >= self.CEILING_RETRY_LAPS:
                    self.scale_ceiling *= 1.05
                    self.clean_laps_at_ceiling = 0
                    limit = self.scale_ceiling * self.CEILING_MARGIN
            self.speed_scale = min(self.speed_scale * self.SPEED_UP_FACTOR, limit)
        else:
            self.speed_scale = max(self.speed_scale * self.WIDE_LAP_FACTOR, self.MIN_SPEED_SCALE)
        self._save_tuning()

    def _adapt_after_crash(self):
        """The current speed is too fast: remember it as the ceiling and slow down."""
        self.scale_ceiling = min(self.scale_ceiling, self.speed_scale)
        self.speed_scale = max(self.speed_scale * self.CRASH_FACTOR, self.MIN_SPEED_SCALE)
        self.clean_laps_at_ceiling = 0
        self._save_tuning()

    def _planned_speed(self, speed_limits, segment_lengths, start_idx, fraction, speed=0.0):
        """Respect upcoming corner limits early enough for the available braking distance.

        The distance covered during the control latency is not available for
        braking, so it is subtracted before computing the reachable speed.
        """
        latency_distance = max(speed, 0.0) * self.BRAKE_LATENCY
        target = self._max_speed()
        distance = 0.0
        idx = start_idx
        remaining = segment_lengths[idx] * (1.0 - fraction)
        braking_horizon = target ** 2 / (2.0 * self.MAX_BRAKING_ACCEL)

        for step in range(len(segment_lengths)):
            limit = float(speed_limits[idx])
            reachable_speed = np.sqrt(
                limit ** 2 + 2.0 * self.MAX_BRAKING_ACCEL * max(distance - latency_distance, 0.0)
            )
            target = min(target, float(reachable_speed))

            distance += remaining
            if distance > braking_horizon + latency_distance:
                break
            idx = (idx + 1) % len(segment_lengths)
            remaining = segment_lengths[idx]
        return target

    @staticmethod
    def _route_geometry(points):
        """Segment lengths, cumulative arc length and right-hand unit normals."""
        segments = np.roll(points, -1, axis=0) - points
        segment_lengths = np.hypot(*segments.T)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)[:-1]))
        right = np.stack([segments[:, 1], -segments[:, 0]], axis=1) / segment_lengths[:, None]
        return segment_lengths, cumulative, right

    @staticmethod
    def _locate_on_route(points, queries):
        """Project many world points at once onto the route.

        Returns arc length s, signed lateral offset (+ = right of the driving
        direction) and distance to the centerline for each query point.
        """
        queries = np.asarray(queries, dtype=float).reshape(-1, 2)
        segments = np.roll(points, -1, axis=0) - points
        length_sq = np.einsum("ij,ij->i", segments, segments)
        relative = queries[:, None, :] - points[None, :, :]
        fraction = np.clip(np.einsum("mnk,nk->mn", relative, segments) / length_sq, 0.0, 1.0)
        offset = relative - fraction[..., None] * segments[None, :, :]
        distance = np.hypot(offset[..., 0], offset[..., 1])
        idx = np.argmin(distance, axis=1)
        rows = np.arange(len(queries))
        segment_lengths = np.sqrt(length_sq)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)[:-1]))
        s = cumulative[idx] + fraction[rows, idx] * segment_lengths[idx]
        right = np.stack([segments[:, 1], -segments[:, 0]], axis=1) / segment_lengths[:, None]
        lateral = np.einsum("mk,mk->m", relative[rows, idx], right[idx])
        return s, lateral, distance[rows, idx]

    def _calibrate_position_scale(self, points, timeout=5.0):
        """Find the factor that turns telemetry positions into track units.

        The right factor is the one for which our distance to the centerline
        matches the simulator's own cross-track error (cte).
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._check_connection()
            if self.position is not None and self.last_cte is not None:
                break
            self.send_controls(0.0, 0.0, 1.0)
            time.sleep(0.05)
        if self.position is None:
            raise TimeoutError("No telemetry position received; cannot calibrate.")
        raw = np.array(self.position) / self.position_scale
        try:
            cte = abs(float(self.last_cte))
        except (TypeError, ValueError):
            cte = 0.0      # no cte in this scene: pick the closest fit
        errors = []
        for k in self.POSITION_SCALE_CANDIDATES:
            _, _, distance = self._locate_on_route(points, raw * k)
            errors.append(abs(float(distance[0]) - cte))
        self.position_scale = self.POSITION_SCALE_CANDIDATES[int(np.argmin(errors))]
        self.position = tuple(raw * self.position_scale)
        if self.last_speed is not None:
            self.last_speed *= self.position_scale
        logging.info("Position scale: telemetry x %.0f (distance-vs-cte errors %s).",
                     self.position_scale, ", ".join(f"{e:.2f}" for e in errors))

    def _map_lidar_scan(self, scan, points, obstacle_map):
        """Convert one lidar sweep to world coordinates and add on-track hits to the map."""
        scan_position, scan_yaw, returns = scan
        try:
            d = np.array([float(p["d"]) for p in returns])
            rx = np.array([float(p["rx"]) for p in returns])
        except (KeyError, TypeError, ValueError):
            return 0, 0, 0
        valid = (d > 0.3) & (d < self.LIDAR_RANGE)
        if not np.any(valid):
            return len(d), 0, 0
        heading = np.deg2rad(scan_yaw)
        origin = np.array(scan_position) + self.LIDAR_FORWARD_OFFSET * np.array(
            [np.sin(heading), np.cos(heading)])
        angle = heading + self.LIDAR_ANGLE_SIGN * np.deg2rad(rx[valid])
        world = origin + d[valid][:, None] * np.stack([np.sin(angle), np.cos(angle)], axis=1)
        _, _, distance = self._locate_on_route(points, world)
        on_track = world[distance < self.TRACK_HALF_WIDTH]
        new_cells = sum(obstacle_map.add(point) for point in on_track)
        return len(d), len(on_track), new_cells

    def _plan_offset(self, obstacle_s, obstacle_lateral, car_s, route_length, current_offset):
        """Choose the lateral offset from the centerline that clears every obstacle ahead.

        Returns (offset, clearance to the closest obstacle in the window,
        whether any obstacle is in the window).
        """
        if len(obstacle_s) == 0:
            return 0.0, np.inf, False
        ds = (obstacle_s - car_s + self.AVOID_BEHIND) % route_length - self.AVOID_BEHIND
        in_window = ds < self.AVOID_HORIZON
        if not np.any(in_window):
            return 0.0, np.inf, False
        lateral = obstacle_lateral[in_window]
        candidates = np.linspace(-self.MAX_PATH_OFFSET, self.MAX_PATH_OFFSET, 41)
        clearance = np.abs(candidates[:, None] - lateral[None, :]).min(axis=1)
        violation = np.clip(self.OBSTACLE_CLEARANCE - clearance, 0.0, None)
        cost = (100.0 * violation
                + 0.3 * np.abs(candidates)                     # prefer the centerline
                + 0.1 * np.abs(candidates - current_offset))   # tie-break: keep the chosen side
        best = int(np.argmin(cost))
        return float(candidates[best]), float(clearance[best]), True

    def _lookahead(self, speed, segment_radius, segment_lengths, start_idx, fraction):
        """Speed-dependent lookahead, shortened before tight corners.

        A long lookahead is stable on straights, but past a corner entry it
        makes pure pursuit cut the corner; so it may extend at most
        LOOKAHEAD_CORNER_FACTOR * R beyond the entry of a corner of radius R.
        """
        lookahead = float(np.clip(self.LOOKAHEAD_MIN + self.LOOKAHEAD_SPEED_GAIN * speed,
                                  self.LOOKAHEAD_MIN, self.LOOKAHEAD_MAX))
        n = len(segment_lengths)
        idx = start_idx
        distance = 0.0
        remaining = segment_lengths[idx] * (1.0 - fraction)
        for _ in range(n):
            if distance >= lookahead:
                break
            limit = distance + self.LOOKAHEAD_CORNER_FACTOR * float(segment_radius[idx])
            lookahead = min(lookahead, max(self.LOOKAHEAD_MIN, limit))
            distance += remaining
            idx = (idx + 1) % n
            remaining = segment_lengths[idx]
        return lookahead

    def _heading_error(self, target, yaw_rad):
        """Angle (rad) between car heading and the target; + means target is to the right."""
        dx = target[0] - self.position[0]
        dz = target[1] - self.position[1]
        lateral = dx * np.cos(yaw_rad) - dz * np.sin(yaw_rad)
        forward = dx * np.sin(yaw_rad) + dz * np.cos(yaw_rad)
        return float(np.arctan2(lateral, forward))

    # ------------------------------------------------------------------
    def drive_autopilot(self, duration=50.0, recorder=None):
        """Follow the simulator centerline (world coordinates + yaw) with pure pursuit.

        The speed is tuned lap after lap (see _adapt_after_lap/_adapt_after_crash).
        If a DatasetRecorder is given, every new camera frame is saved with the
        controls sent for it; frames from the seconds before a crash are dropped.
        """
        self._load_tuning()
        print("Resetting car to the track start...")
        self.send_now(json.dumps({"msg_type": "reset_car"}))
        time.sleep(1.0)

        self._wait_for_route()
        points = np.array([self.track_nodes[i] for i in range(self.total_nodes)], dtype=float)
        n = len(points)
        if n < 3:
            raise ValueError(f"Track centerline needs at least 3 nodes; received {n}.")
        segment_lengths = np.hypot(*(np.roll(points, -1, axis=0) - points).T)
        if np.any(segment_lengths <= 1e-6):
            raise ValueError("Track centerline contains duplicate or zero-length segments.")
        speed_limits = self._segment_speed_limits(points, segment_lengths)
        segment_lengths, cumulative_s, right_normals = self._route_geometry(points)
        segment_radius = 1.0 / np.maximum(self._segment_curvature(points), 1e-6)
        route_length = float(segment_lengths.sum())

        # Obstacles live in world coordinates; their (s, lateral) position along
        # the route is recomputed whenever the map changes.
        self._calibrate_position_scale(points)
        obstacle_map = ObstacleMap(cell_size=self.OBSTACLE_CELL_SIZE, min_hits=3)
        obstacle_s = obstacle_lateral = obstacle_xy = np.empty(0)
        last_lidar_seq = self.lidar_seq
        lidar_reported = False
        handled_collisions = len(self.collisions)
        path_offset = 0.0

        steering = 0.0
        start = time.monotonic()
        last_progress_time = start
        last_progress_pos = None
        off_track_since = None
        nearest = None
        route_direction_checked = False
        next_status_log = start
        sensor_timeout = 0.5
        speed_integral = 0.0
        last_loop_time = start

        # Lap bookkeeping for the adaptive speed search
        previous_nearest = None
        lap_progress = 0
        lap_max_offset = 0.0
        lap_start = start
        laps = 0
        best_lap = None

        # Frames wait here DISCARD_BEFORE_CRASH seconds before being written,
        # so the approach to a crash never ends up in the dataset.
        pending_frames = deque()
        last_recorded_seq = self.image_seq

        print(f"Autopilot driving for {duration:g} seconds "
              f"(speed_scale={self.speed_scale:.2f}).")
        while time.monotonic() - start < duration:
            self._check_connection()
            now = time.monotonic()
            dt = min(now - last_loop_time, 0.1)
            last_loop_time = now

            fresh = (self.position is not None and self.yaw is not None
                     and self.last_position_time is not None
                     and now - self.last_position_time <= sensor_timeout)
            if not fresh:
                self.send_controls(0.0, 0.0, 1.0)   # no coordinates -> stop, never guess
                time.sleep(0.02)
                continue

            pos = np.array(self.position)
            speed = self.last_speed if self.last_speed is not None else 0.0
            if not route_direction_checked:
                initial_projection = self._project_to_route(points, pos)
                if initial_projection is None:
                    raise RuntimeError("Could not determine the initial track direction.")
                _, segment_idx, _, _ = initial_projection
                tangent = points[(segment_idx + 1) % n] - points[segment_idx]
                yaw_rad = np.deg2rad(self.yaw)
                forward = np.array([np.sin(yaw_rad), np.cos(yaw_rad)])
                if float(np.dot(tangent, forward)) < 0.0:
                    points = points[::-1].copy()
                    segment_lengths, cumulative_s, right_normals = self._route_geometry(points)
                    segment_radius = 1.0 / np.maximum(self._segment_curvature(points), 1e-6)
                    speed_limits = self._segment_speed_limits(points, segment_lengths)
                    logging.info("Reversed track-node order to match the car's starting heading.")
                route_direction_checked = True

            # --- Where am I on the track? (search near last index to avoid
            #     jumping to a different part of the maze that passes close by)
            projection = self._project_to_route(points, pos, nearest)
            if projection is None:
                raise RuntimeError("Could not project the vehicle onto the track centerline.")
            dist_to_center, nearest, segment_fraction, _ = projection
            if dist_to_center > self.OFF_TRACK_DISTANCE and nearest is not None:
                projection = self._project_to_route(points, pos)
                dist_to_center, nearest, segment_fraction, _ = projection
            car_s = cumulative_s[nearest] + segment_fraction * segment_lengths[nearest]
            car_lateral = float(np.dot(pos - points[nearest], right_normals[nearest]))

            # --- Obstacle map: new lidar sweep and collisions reported by the sim
            map_changed = False
            if self.lidar_seq != last_lidar_seq and self.lidar_scan is not None:
                last_lidar_seq = self.lidar_seq
                total_returns, on_track, new_cells = self._map_lidar_scan(
                    self.lidar_scan, points, obstacle_map)
                map_changed = new_cells > 0
                if not lidar_reported:
                    logging.info("Lidar working: %d returns in first sweep, %d on the track.",
                                 total_returns, on_track)
                    lidar_reported = True
            while handled_collisions < len(self.collisions):
                name, hit_pos, _ = self.collisions[handled_collisions]
                handled_collisions += 1
                # Not added to the obstacle map: the lidar already maps the real
                # object, while a point guessed from the car pose can land in
                # the free passage and block it for good.
                old_scale = self.speed_scale
                self._adapt_after_crash()
                speed_limits = self._segment_speed_limits(points, segment_lengths)
                logging.warning("Collision with %r at (%.1f, %.1f); speed_scale %.2f -> %.2f.",
                                name, hit_pos[0], hit_pos[1], old_scale, self.speed_scale)
            if map_changed:
                obstacle_xy = obstacle_map.confirmed()
                if len(obstacle_xy):
                    obstacle_s, obstacle_lateral, _ = self._locate_on_route(points, obstacle_xy)
                else:
                    obstacle_s = obstacle_lateral = np.empty(0)

            # --- Avoidance: shift the followed path sideways around obstacles ahead
            desired_offset, clearance, obstacle_ahead = self._plan_offset(
                obstacle_s, obstacle_lateral, car_s, route_length, path_offset)
            path_offset += float(np.clip(desired_offset - path_offset,
                                         -self.OFFSET_RATE, self.OFFSET_RATE))
            tracking_error = abs(car_lateral - path_offset)

            # --- Lap counting: accumulate signed node progress along the route
            if previous_nearest is not None:
                step = (nearest - previous_nearest) % n
                if step > n // 2:
                    step -= n
                lap_progress += step
            previous_nearest = nearest
            # Judge laps by how well the planned path was tracked, so that a
            # deliberate swerve around an obstacle does not count as "too wide".
            lap_max_offset = max(lap_max_offset, tracking_error)
            if lap_progress >= n:
                lap_time = now - lap_start
                laps += 1
                best_lap = lap_time if best_lap is None else min(best_lap, lap_time)
                old_scale = self.speed_scale
                self._adapt_after_lap(lap_max_offset)
                speed_limits = self._segment_speed_limits(points, segment_lengths)
                logging.info("Lap %d: %.2fs (best %.2fs), max off-center %.2fm, "
                             "speed_scale %.2f -> %.2f",
                             laps, lap_time, best_lap, lap_max_offset,
                             old_scale, self.speed_scale)
                lap_progress -= n
                lap_max_offset = 0.0
                lap_start = now

            # --- Recovery: stuck or far off the track -> reset to start
            if dist_to_center > self.OFF_TRACK_DISTANCE:
                off_track_since = off_track_since or now
            else:
                off_track_since = None
            if last_progress_pos is None or np.hypot(*(pos - last_progress_pos)) > 2.0:
                last_progress_pos = pos.copy()
                last_progress_time = now
            # A car can rotate through a tight corner without moving 0.5 m in
            # a straight line. Do not reset it for that; require near-zero
            # simulator speed as well as no positional progress.
            stuck = (now - last_progress_time > self.STUCK_SECONDS
                     and abs(speed) < self.STUCK_MIN_SPEED)
            off = off_track_since is not None and now - off_track_since > 1.5
            if stuck or off:
                old_scale = self.speed_scale
                self._adapt_after_crash()
                speed_limits = self._segment_speed_limits(points, segment_lengths)
                logging.warning("Car %s; dropping %d recent frames, speed_scale %.2f -> %.2f, "
                                "resetting to track start.", "stuck" if stuck else "off track",
                                len(pending_frames), old_scale, self.speed_scale)
                pending_frames.clear()
                self.send_now(json.dumps({"msg_type": "reset_car"}))
                time.sleep(1.0)
                steering, nearest, speed_integral, path_offset = 0.0, None, 0.0, 0.0
                last_progress_pos, last_progress_time = None, time.monotonic()
                off_track_since = None
                previous_nearest, lap_progress, lap_max_offset = None, 0, 0.0
                lap_start = last_loop_time = time.monotonic()
                last_recorded_seq = self.image_seq
                continue

            # --- Steering: pure pursuit toward a speed-dependent lookahead point
            yaw_rad = np.deg2rad(self.yaw)
            lookahead = self._lookahead(speed, segment_radius, segment_lengths,
                                        nearest, segment_fraction)
            target = self._point_ahead(points, nearest, segment_fraction, lookahead,
                                       segment_lengths)
            if path_offset != 0.0:
                beyond = self._point_ahead(points, nearest, segment_fraction, lookahead + 0.5,
                                           segment_lengths)
                tangent = beyond - target
                norm = float(np.hypot(*tangent))
                if norm > 1e-6:
                    target = target + path_offset * np.array([tangent[1], -tangent[0]]) / norm
            alpha = self._heading_error(target, yaw_rad)
            target_steering = float(np.clip(self.STEER_SIGN * self.STEER_GAIN * alpha, -1.0, 1.0))

            # --- Speed: obey curvature limits and brake before upcoming corners
            target_speed = self._planned_speed(
                speed_limits, segment_lengths, nearest, segment_fraction, speed
            )
            if obstacle_ahead and clearance < self.OBSTACLE_CLEARANCE:
                target_speed = min(target_speed, self.NO_GAP_SPEED)
            elif obstacle_ahead and clearance < 2.0 * self.OBSTACLE_CLEARANCE:
                target_speed = min(target_speed, self.AVOID_MAX_SPEED)

            # --- Emergency: an obstacle straight ahead inside the braking distance
            emergency = False
            if len(obstacle_xy):
                rel = obstacle_xy - pos
                ahead = rel[:, 0] * np.sin(yaw_rad) + rel[:, 1] * np.cos(yaw_rad)
                side = rel[:, 0] * np.cos(yaw_rad) - rel[:, 1] * np.sin(yaw_rad)
                stop_distance = speed ** 2 / (2.0 * self.MAX_BRAKING_ACCEL) + 0.5
                emergency = bool(np.any((ahead > 0.0) & (ahead < stop_distance)
                                        & (np.abs(side) < self.CAR_HALF_WIDTH)))
            if emergency:
                target_speed = 0.0
            error = target_speed - speed
            # The integral term supplies the extra throttle needed to hold
            # higher speeds; it only charges while we are not braking.
            if error > -2.0:
                speed_integral = float(np.clip(speed_integral + self.THROTTLE_KI * error * dt,
                                               0.0, 0.5))
                throttle = float(np.clip(0.25 + self.THROTTLE_KP * error + speed_integral,
                                         0.0, self.MAX_THROTTLE))
                brake = 0.0
            else:
                throttle = 0.0
                brake = float(np.clip(-self.BRAKE_KP * error, 0.0, 1.0))

            if speed < min(target_speed, 2.0):
                throttle = max(throttle, 0.3)
            if emergency:
                throttle, brake, speed_integral = 0.0, 1.0, 0.0

            # --- Smooth steering (rate-limited)
            steering += float(np.clip(target_steering - steering,
                                      -self.MAX_STEER_RATE, self.MAX_STEER_RATE))

            self.send_controls(steering, throttle, brake)

            # --- Dataset: pair each new camera frame with the controls just sent
            if recorder is not None:
                if self.image_seq != last_recorded_seq and self.last_image is not None:
                    last_recorded_seq = self.image_seq
                    pending_frames.append((now, self.last_image, {
                        "steering": round(steering, 4),
                        "throttle": round(throttle, 4),
                        "brake": round(brake, 4),
                        "speed": round(speed, 3),
                        "target_speed": round(target_speed, 3),
                        "off_center": round(dist_to_center, 3),
                        "lap": laps,
                        "speed_scale": round(self.speed_scale, 3),
                        "timestamp": round(now - start, 3),
                    }))
                while pending_frames and now - pending_frames[0][0] > self.DISCARD_BEFORE_CRASH:
                    _, image, labels = pending_frames.popleft()
                    recorder.write(image, labels)

            if now >= next_status_log:
                logging.info(
                    "Autopilot: speed=%.2f target=%.2f node=%d/%d off-center=%.2fm "
                    "(sim cte=%s) path-offset=%.2fm obstacles=%d clearance=%.2f%s "
                    "steer=%.2f throttle=%.2f scale=%.2f frames=%d",
                    speed, target_speed, nearest, n, dist_to_center, self.last_cte,
                    path_offset, len(obstacle_xy), clearance,
                    " EMERGENCY-BRAKE" if emergency else "", steering, throttle,
                    self.speed_scale, recorder.count if recorder else 0,
                )
                next_status_log = now + 3.0
            time.sleep(0.02)

        # Normal end of the run: the last frames are fine, keep them.
        if recorder is not None:
            for _, image, labels in pending_frames:
                recorder.write(image, labels)
        logging.info("Run finished: %d laps, best lap %s, final speed_scale %.2f.",
                     laps, f"{best_lap:.2f}s" if best_lap else "n/a", self.speed_scale)


###########################################
## Make a client and connect to the simulator

def test_clients():
    logging.basicConfig(level=logging.INFO)

    host = "127.0.0.1"
    port = 9091
    num_clients = 1
    clients = []
    time_to_drive = 600.0   # 10 minutes of driving / recording
    record_dataset = False  # set to True to save camera frames + labels
    recorder = None

    try:
        for _ in range(num_clients):
            clients.append(SimpleClient(address=(host, port)))

        print(f"Waiting for the simulator to load {clients[0].scene_name}...")
        load_deadline = time.monotonic() + 30.0
        exit_sent = False
        # Only accept a car loaded in the scene WE requested: if the simulator
        # was already inside another scene, leave it so the menu lets us
        # load the forced one.
        while (not (clients[0].car_loaded and clients[0].scene_load_requested)
               and time.monotonic() < load_deadline):
            c = clients[0]
            if c.car_loaded and not c.scene_load_requested and not exit_sent:
                # The simulator only announces the scene menu to a new
                # connection, so leave the scene and reconnect.
                logging.info("Simulator is already in a scene; leaving it to load %s.",
                             c.scene_name)
                c.send_now(json.dumps({"msg_type": "exit_scene"}))
                time.sleep(1.5)
                c.stop()
                time.sleep(0.5)
                clients[0] = c = SimpleClient(address=(host, port))
                exit_sent = True
            if (c.scene_selection_time is not None
                    and not c.scene_names_received
                    and not c.scene_load_requested
                    and time.monotonic() - c.scene_selection_time >= 1.0):
                logging.warning("Simulator did not return scene names; requesting %s directly.",
                                c.scene_name)
                c.scene_load_requested = True
                c.scene_load_requested_at = time.monotonic()
                c.send_now(json.dumps({"msg_type": "load_scene", "scene_name": c.scene_name}))
            if (c.scene_load_requested
                    and c.scene_load_requested_at is not None
                    and not c.car_configured
                    and time.monotonic() - c.scene_load_requested_at >= 2.0):
                logging.info("Scene load fallback delay elapsed; sending configuration.")
                c.send_config()
            time.sleep(0.1)
        if not clients[0].car_loaded:
            raise TimeoutError(
                "The simulator did not report car_loaded within 30 seconds. "
                "Check that the simulator is open and has finished loading."
            )

        print("Car loaded. Starting autopilot.")
        if record_dataset:
            recorder = DatasetRecorder()
        clients[0].drive_autopilot(duration=time_to_drive, recorder=recorder)
        time.sleep(1.0)
    finally:
        if recorder is not None:
            recorder.close()
        print("Waiting for message loops to stop.")
        for client in clients:
            client.stop()
        print("Clients stopped.")


if __name__ == "__main__":
    test_clients()
