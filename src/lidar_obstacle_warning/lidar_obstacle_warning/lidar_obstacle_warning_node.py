#!/usr/bin/env python3
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, Tuple as TTuple

import os
from datetime import datetime
from pathlib import Path
import colorsys

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3, TransformStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster
from rclpy.node import Node
from std_msgs.msg import Int32

# QoS imports for transient-local map subscription
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

# Optional PIL for debug image generation (only used if debug enabled)
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


def _hsv_to_rgb_tuple(h: float, s: float = 1.0, v: float = 1.0, alpha: int = 220):
    """Return an RGBA tuple (0-255) from HSV values (h in [0,1])."""
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255), int(alpha))


def convex_hull(points: np.ndarray) -> List[TTuple[float, float]]:
    """
    Monotone chain convex hull. points: (N,2) array
    Returns list of (x,y) tuples of hull in CCW order. If <=2 points, returns them.
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    pts = sorted(set(pts))
    if len(pts) <= 1:
        return pts
    # lower
    lower: List[TTuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2:
            (x1, y1) = lower[-2]
            (x2, y2) = lower[-1]
            cross = (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(p)
    # upper
    upper: List[TTuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2:
            (x1, y1) = upper[-2]
            (x2, y2) = upper[-1]
            cross = (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)
            if cross <= 0:
                upper.pop()
            else:
                break
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull


class ObstacleTracker(Node):
    """
    Obstacle detection node with debug visualization levels:
      debug_level = 0 -> no debug (default)
      debug_level = 1 -> save images when we have both pose and scan (pose-triggered and scan-updates with pose)
      debug_level = 2 -> same as 1 plus save visualizations even when only scan is available

    Images include scan number in filename and a color/symbol legend.

    New features:
    - min_point_distance: filter out points nearer than this (default 0.2m).
    - /debug_image_request (Int32): when a positive number N is published here, the node will save N debug images
      (decrementing the counter each time an image is written). This allows ad-hoc collection without enabling
      continuous per-scan debug.
    """

    def __init__(self):
        super().__init__('obstacle_tracker')

        # Parameters (tweak these to tune perf/behavior)
        self.scan_skip = 1  # process every Nth scan
        self.range_downsample = 1  # downsample LIDAR ranges (1 = keep all, 2 = every other)
        self.cluster_min_points = 5
        self.cluster_min_distance = 0.04  # max distance between consecutive points in a cluster
        self.cluster_max_diameter = 0.3  # max distance between two furthest points in a cluster (None = no limit)
        self.close_threshold = 0.5
        # NEW: maximum allowed jump (meters) between the previously tracked obstacle and the closest cluster.
        # If exceeded, tracking will be stopped from the scan callback.
        self.scan_tracking_stop_distance = 0.8

        # New: minimum range to keep (filter out points that are too close)
        self.min_point_distance =  0.2  # meters, default 0.2

        # Debugging / visualization parameters
        # Default debug level changed to 0 to avoid saving images unless explicitly requested.
        # 0 = off, 1 = visualize when we have pose and scan, 2 = also visualize for scan-only cases
        self.debug_level = 0 # default now 0 (no images)
        try:
            repo_debug_default = Path(__file__).resolve().parents[1] / 'debug'
        except Exception:
            repo_debug_default = Path('/tmp/obstacle_debug') 
        env_folder = os.environ.get('OBSTACLE_DEBUG_FOLDER')
        self.debug_folder = Path(env_folder) if env_folder else repo_debug_default
        # Pixels per map cell in final image (bigger -> more detailed)
        self.debug_px_per_cell = 4
        # Additional meter margin around combined map+scan extents in debug image
        self.debug_margin_m = 1.0
        # Max image dimension to prevent enormous files
        self.debug_max_dim = 2000

        # Background color for debug canvases (RGBA). Default is a neutral gray background.
        default_bg = (230, 230, 230, 255)
        env_bg = os.environ.get('OBSTACLE_DEBUG_BG')
        if env_bg:
            try:
                parts = [int(p) for p in env_bg.split(',')]
                if len(parts) == 3:
                    parts.append(255)
                if len(parts) == 4:
                    default_bg = tuple(parts)
            except Exception:
                self.get_logger().warning(f'Invalid OBSTACLE_DEBUG_BG value: {env_bg}; using default {default_bg}')
        self.debug_bg_color = default_bg

        # Counter for forced debug images requested via topic
        self.forced_debug_remaining = 0

        # enable debug only if level > 0 and PIL is available
        if self.debug_level > 0 and not PIL_AVAILABLE:
            self.get_logger().warning('Pillow (PIL) is not available; debug images will be disabled (set OBSTACLE_DEBUG_LEVEL=0 to avoid this).')
            self.debug_level = 0

        if self.debug_level > 0:
            try:
                self.debug_folder.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.get_logger().warning(f'Failed to create debug folder {self.debug_folder}: {e}')
                self.debug_level = 0
        else:
            # ensure folder exists only when needed by forced requests
            try:
                self.debug_folder.mkdir(parents=True, exist_ok=True)
            except Exception:
                # ignore; will log on first save attempt
                pass

        # Log current debug startup state so user understands why images are/aren't saved
        self.get_logger().info(f'ObstacleTracker starting: debug_level={self.debug_level}, debug_folder="{self.debug_folder}", min_point_distance={self.min_point_distance:.2f} m')

        # Subscriptions / publishers
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # New: subscription for ad-hoc debug image requests (Int32: number of images to record)
        self.create_subscription(Int32, '/debug_image_request', self._debug_image_request_callback, 10)

        qos_map = QoSProfile(depth=1)
        qos_map.reliability = ReliabilityPolicy.RELIABLE
        qos_map.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos_map.history = HistoryPolicy.KEEP_LAST
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_map)

        self.create_subscription(Vector3, '/robot_pose_raw', self.pose_callback, 10)
        self.obstacle_pub = self.create_publisher(PoseStamped, '/obstacle_relative', 10)
        self.close_pub = self.create_publisher(Int32, '/emergency_break', 10)
        self.marker_pub = self.create_publisher(Marker, '/obstacle_candidate_marker', 10)

        # TF broadcaster (event-driven: publish on each new pose)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Cached state
        self.latest_scan: Optional[LaserScan] = None
        self.latest_map_msg: Optional[OccupancyGrid] = None
        self.map_np: Optional[np.ndarray] = None  # 2D array shape (height, width)
        self.map_info = {}
        self.current_pose = None  # dict with x,y,theta (robot pose in map frame)
        self.tracked_obstacle = None
        self.tracking = False

        # Helpers
        self.scan_counter = 0
        self.lock = threading.Lock()

        # Threadpool for heavy compute (vectorized processing)
        self.pool = ThreadPoolExecutor(max_workers=2)

        # Reused message objects to avoid allocations
        self._pose_msg = PoseStamped()
        self._pose_msg.header.frame_id = 'base_link'
        self._pose_msg.pose.position.z = 0.0
        self._pose_msg.pose.orientation.w = 1.0

        self._marker_msg = Marker()
        # header.frame_id will be set when publishing (map if current_pose available)
        self._marker_msg.ns = 'obstacle'
        self._marker_msg.id = 0
        self._marker_msg.type = Marker.SPHERE
        self._marker_msg.action = Marker.ADD
        self._marker_msg.scale.x = 0.2
        self._marker_msg.scale.y = 0.2
        self._marker_msg.scale.z = 0.2
        self._marker_msg.color.a = 1.0
        self._marker_msg.color.r = 1.0
        self._marker_msg.color.g = 0.0
        self._marker_msg.color.b = 0.0
        self._marker_msg.pose.orientation.w = 1.0

        # Precreate delete marker copy (frame will be set dynamically when publishing)
        self._delete_marker = Marker()
        self._delete_marker.ns = 'obstacle'
        self._delete_marker.id = 0
        self._delete_marker.action = Marker.DELETE
        self._delete_marker.type = Marker.SPHERE

    #
    # New: topic callback to request N debug images be saved
    #
    def _debug_image_request_callback(self, msg: Int32):
        try:
            n = int(msg.data)
        except Exception:
            self.get_logger().warning(f'Invalid debug_image_request value: {msg.data}')
            return
        if n <= 0:
            with self.lock:
                self.forced_debug_remaining = 0
            self.get_logger().info('Cleared forced debug image requests (set to 0).')
        else:
            with self.lock:
                self.forced_debug_remaining = n
            # ensure debug folder exists when forced images will be saved
            try:
                self.debug_folder.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.get_logger().warning(f'Failed to create debug folder {self.debug_folder}: {e}')
            self.get_logger().info(f'Request received: will save {n} debug images (or until counter reaches 0).')

    #
    # Callbacks
    #
    def scan_callback(self, msg: LaserScan):
        # Keep the latest scan
        self.latest_scan = msg
        # increment scan counter (used in filenames)
        with self.lock:
            self.scan_counter += 1
            current_scan_num = int(self.scan_counter)
            forced_remaining = int(self.forced_debug_remaining)

        # If we have a known robot pose, re-publish the map->base_link transform
        # stamped with the scan's header.stamp so TF consumers can transform this scan.
        with self.lock:
            pose_for_tf = None if self.current_pose is None else self.current_pose.copy()

        if pose_for_tf is not None:
            # publish transform stamped at scan time (use scan.header.stamp if present)
            stamp = None
            try:
                stamp = msg.header.stamp
            except Exception:
                stamp = None
            self._send_transform(pose_for_tf, stamp=stamp)

        # Throttle processing: only handle every self.scan_skip scans
        if (current_scan_num - 1) % self.scan_skip != 0:
            return

        try:
            # pass scan number into processing for debug decisions
            self.pool.submit(self._process_scan, msg, current_scan_num)
        except Exception:
            self._process_scan(msg, current_scan_num)

    def _process_scan(self, scan: LaserScan, scan_num: int):
        # Update location of currently tracked obstacle (if any)
        with self.lock:
            tracking = self.tracking
            tracked_exists = self.tracked_obstacle is not None
            current_pose_copy = None if self.current_pose is None else self.current_pose.copy()
            forced_remaining = int(self.forced_debug_remaining)

        if not tracking or not tracked_exists:
            # If debug level 2 or forced images remain, we still save scan-only visualizations even when not tracking
            if (self.debug_level >= 2) or (forced_remaining > 0):
                # Create visualization even if no tracking; we may not have a pose => pass pose at origin
                pose_for_viz = current_pose_copy if current_pose_copy is not None else {'x': 0.0, 'y': 0.0, 'theta': 0.0}
                pts, _ = self.scan_to_points(scan, downsample=self.range_downsample, return_indices=True)
                if pts.size:
                    self._save_debug_image(scan, pts, pts, [], [], pose_for_viz, label='scan_only', scan_num=scan_num, pose_present=(current_pose_copy is not None))
            return

        # Use only the scan for scan-time updates (do NOT consult the map/pose here).
        pts, _ = self.scan_to_points(scan, downsample=self.range_downsample, return_indices=True)
        if pts.size == 0:
            # No points in this scan — keep existing tracking unchanged.
            return

        # Cluster directly on raw scan points (scan-ordered). These points are
        # expressed in the robot frame (base_link).
        accepted_clusters, rejected_clusters = self.cluster_scan_ordered(
            pts,
            min_points=self.cluster_min_points,
            max_gap=self.cluster_min_distance,
            max_diameter=self.cluster_max_diameter,
        )

        with self.lock:
            forced_remaining = int(self.forced_debug_remaining)

        # Save images depending on debug level or forced counter:
        if (self.debug_level >= 1 and current_pose_copy is not None) or (forced_remaining > 0):
            pose_for_viz = current_pose_copy if current_pose_copy is not None else {'x': 0.0, 'y': 0.0, 'theta': 0.0}
            label = 'scan_update' if accepted_clusters else 'scan_update_no_clusters'
            self._save_debug_image(scan, pts, pts, accepted_clusters, rejected_clusters, pose_for_viz, label=label, scan_num=scan_num, pose_present=(current_pose_copy is not None))

        if not accepted_clusters:
            # No clusters found in this scan. Do not clear the tracked obstacle here.
            return

        # Find the cluster whose centroid is closest to the previously tracked obstacle
        with self.lock:
            prev = np.array([self.tracked_obstacle['x'], self.tracked_obstacle['y']])
        centroids = [np.mean(c, axis=0) for c in accepted_clusters]
        dists = [np.linalg.norm(c - prev) for c in centroids]
        best_idx = int(np.argmin(dists))
        best_centroid = centroids[best_idx]
        best_dist = dists[best_idx]

        # NEW: stop tracking if the closest cluster is too far from the previously tracked obstacle.
        if best_dist > self.scan_tracking_stop_distance:
            with self.lock:
                self.tracking = False
                self.tracked_obstacle = None
            self._publish_delete_marker()
            return

        with self.lock:
            self.tracked_obstacle = {'x': float(best_centroid[0]), 'y': float(best_centroid[1])}

        # Publish the obstacle in the robot frame for scan-based updates (avoid using map/pose)
        self.publish_obstacle(self.tracked_obstacle)
        # When updating from scans we should publish the marker in base_link (robot frame)
        self.publish_marker(self.tracked_obstacle, use_map=False)

    def map_callback(self, msg: OccupancyGrid):
        self.latest_map_msg = msg
        info = msg.info
        width = info.width
        height = info.height
        # msg.data may be bytes or list: handle both
        if isinstance(msg.data, (bytes, bytearray)):
            arr = np.frombuffer(msg.data, dtype=np.int8)
        else:
            arr = np.array(msg.data, dtype=np.int8)
        try:
            arr = arr.reshape((height, width))
        except Exception:
            arr = np.array(msg.data, dtype=np.int8).reshape((height, width))
        self.map_np = arr

        # Extract origin position and yaw (orientation) and store in map_info.
        ox = info.origin.position.x
        oy = info.origin.position.y
        q = info.origin.orientation
        # Convert quaternion -> yaw robustly (handles general quaternion)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.map_info = {
            'res': info.resolution,
            'width': width,
            'height': height,
            'ox': ox,
            'oy': oy,
            'yaw': yaw,
        }
        self.get_logger().debug(f"Received map: res={info.resolution} ox={ox} oy={oy} yaw={yaw} size={(width,height)}")

    def pose_callback(self, msg: Vector3):
        """
        Event-driven TF publishing: publish map->base_link transform immediately when a new
        robot pose is received on /robot_pose_raw.
        """
        pose = {'x': float(msg.x), 'y': float(msg.y), 'theta': float(msg.z)}
        with self.lock:
            self.tracking = False
            self.tracked_obstacle = None
            self.current_pose = pose

        # Publish TF once right away for this new pose (event-driven)
        self._send_transform(pose)

        if self.latest_scan is None or self.latest_map_msg is None:
            return

        # For pose-triggered processing we want to use the latest scan number for filename.
        with self.lock:
            scan_num = int(self.scan_counter)
            forced_remaining = int(self.forced_debug_remaining)

        try:
            self.pool.submit(self._process_pose_trigger, self.latest_scan, pose, scan_num)
        except Exception:
            self._process_pose_trigger(self.latest_scan, pose, scan_num)

    def _process_pose_trigger(self, scan: LaserScan, pose: dict, scan_num: int):
        pts, idxs = self.scan_to_points(scan, downsample=self.range_downsample, return_indices=True)
        if pts.size == 0:
            # no points -> ensure marker removed
            self._publish_delete_marker()
            if (self.debug_level >= 1) or (int(self.forced_debug_remaining) > 0):
                self._save_debug_image(scan, pts, pts, [], [], pose, label='pose_trigger_no_points', scan_num=scan_num, pose_present=True)
            return

        filtered_pts = self.filter_free_space(pts, pose, idxs)
        if filtered_pts.size == 0:
            # no valid points -> ensure marker removed
            self._publish_delete_marker()
            if (self.debug_level >= 1) or (int(self.forced_debug_remaining) > 0):
                self._save_debug_image(scan, pts, filtered_pts, [], [], pose, label='pose_trigger_no_filtered', scan_num=scan_num, pose_present=True)
            return

        accepted_clusters, rejected_clusters = self.cluster_scan_ordered(
            filtered_pts,
            min_points=self.cluster_min_points,
            max_gap=self.cluster_min_distance,
            max_diameter=self.cluster_max_diameter,
        )
        if not accepted_clusters:
            # no clusters -> ensure marker removed
            self._publish_delete_marker()
            if (self.debug_level >= 1) or (int(self.forced_debug_remaining) > 0):
                self._save_debug_image(scan, pts, filtered_pts, accepted_clusters, rejected_clusters, pose, label='pose_trigger_no_clusters', scan_num=scan_num, pose_present=True)
            return

        largest = max(accepted_clusters, key=lambda c: c.shape[0])
        centroid = np.mean(largest, axis=0)

        with self.lock:
            self.tracked_obstacle = {'x': float(centroid[0]), 'y': float(centroid[1])}
            self.tracking = True
            self.current_pose = pose

        self.publish_obstacle(self.tracked_obstacle)
        self.publish_marker(self.tracked_obstacle)

        if (self.debug_level >= 1) or (int(self.forced_debug_remaining) > 0):
            self._save_debug_image(scan, pts, filtered_pts, accepted_clusters, rejected_clusters, pose, label='pose_trigger_ok', scan_num=scan_num, pose_present=True)

    #
    # Processing utilities (vectorized)
    #
    def scan_to_points(self, scan: LaserScan, downsample: int = 1, return_indices: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        ranges = np.array(scan.ranges, dtype=np.float32)
        n = ranges.size
        if n == 0:
            return (np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)) if return_indices else np.empty((0, 2), dtype=np.float32)

        angles = scan.angle_min + np.arange(n, dtype=np.float32) * scan.angle_increment
        valid_mask = (ranges > scan.range_min) & (ranges < scan.range_max) & np.isfinite(ranges)

        # New: filter out points that are too close
        if self.min_point_distance is not None and self.min_point_distance > 0.0:
            valid_mask = valid_mask & (ranges >= float(self.min_point_distance))

        if downsample > 1:
            stride_mask = (np.arange(n) % downsample) == 0
            valid_mask = valid_mask & stride_mask

        if not np.any(valid_mask):
            return (np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)) if return_indices else np.empty((0, 2), dtype=np.float32)

        sel_idx = np.nonzero(valid_mask)[0]
        sel_ranges = ranges[sel_idx]
        sel_angles = angles[sel_idx]

        px = sel_ranges * np.cos(sel_angles)
        py = sel_ranges * np.sin(sel_angles)
        pts = np.stack((px, py), axis=-1).astype(np.float32)

        if return_indices:
            return pts, sel_idx.astype(np.int32)
        else:
            return pts

    def filter_free_space(self, points: np.ndarray, pose: dict, indices: Optional[np.ndarray] = None) -> np.ndarray:
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        if self.map_np is None or self.latest_map_msg is None or not self.map_info:
            return points

        res = self.map_info['res']
        width = self.map_info['width']
        height = self.map_info['height']
        ox = self.map_info['ox']
        oy = self.map_info['oy']
        yaw = self.map_info.get('yaw', 0.0)

        ct = math.cos(pose['theta'])
        st = math.sin(pose['theta'])

        # scan points in robot frame -> transform to world/map frame (mx,my)
        px = points[:, 0].astype(np.float64)
        py = points[:, 1].astype(np.float64)
        mx = pose['x'] + ct * px - st * py
        my = pose['y'] + st * px + ct * py

        # Translate world coordinates into map-local coordinates, taking into account map origin rotation.
        dx = mx - ox
        dy = my - oy
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        map_x = cos_y * dx + sin_y * dy
        map_y = -sin_y * dx + cos_y * dy

        cx = np.floor(map_x / res).astype(np.int32)
        cy = np.floor(map_y / res).astype(np.int32)

        in_bounds = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)
        if not np.any(in_bounds):
            return np.empty((0, 2), dtype=np.float32)

        valid_cx = cx[in_bounds]
        valid_cy = cy[in_bounds]

        occ_vals = self.map_np[valid_cy, valid_cx]
        free_mask = in_bounds.copy()
        free_mask[in_bounds] = (occ_vals == 0)

        if not np.any(free_mask):
            return np.empty((0, 2), dtype=np.float32)

        filtered_points = points[free_mask]
        if indices is not None and indices.size == points.shape[0]:
            filtered_indices = indices[free_mask]
            order = np.argsort(filtered_indices)
            filtered_points = filtered_points[order]

        return filtered_points

    def cluster_scan_ordered(self,
                             points: np.ndarray,
                             min_points: int = 3,
                             max_gap: float = 0.1,
                             max_diameter: Optional[float] = None):
        """
        Cluster points in scan order. Returns a tuple:
         (accepted_clusters, rejected_clusters)
        """
        if points.size == 0:
            return [], []

        accepted = []
        rejected = []
        current = [points[0]]
        for i in range(1, points.shape[0]):
            p_prev = points[i - 1]
            p_cur = points[i]
            if np.linalg.norm(p_cur - p_prev) <= max_gap:
                current.append(p_cur)
            else:
                if len(current) >= min_points:
                    cluster_arr = np.array(current)
                    if max_diameter is None:
                        accepted.append(cluster_arr)
                    else:
                        k = cluster_arr.shape[0]
                        if k <= 1:
                            diam = 0.0
                        else:
                            diffs = cluster_arr[:, None, :] - cluster_arr[None, :, :]
                            dists = np.hypot(diffs[..., 0], diffs[..., 1])
                            diam = float(dists.max())
                        if diam <= max_diameter:
                            accepted.append(cluster_arr)
                        else:
                            rejected.append(cluster_arr)
                current = [p_cur]
        if len(current) >= min_points:
            cluster_arr = np.array(current)
            if max_diameter is None:
                accepted.append(cluster_arr)
            else:
                k = cluster_arr.shape[0]
                if k <= 1:
                    diam = 0.0
                else:
                    diffs = cluster_arr[:, None, :] - cluster_arr[None, :, :]
                    dists = np.hypot(diffs[..., 0], diffs[..., 1])
                    diam = float(dists.max())
                if diam <= max_diameter:
                    accepted.append(cluster_arr)
                else:
                    rejected.append(cluster_arr)
        return accepted, rejected

    #
    # Publishing (reusing message objects)
    #
    def publish_obstacle(self, obs: dict):
        self._pose_msg.header.stamp = self.get_clock().now().to_msg()
        self._pose_msg.pose.position.x = float(obs['x'])
        self._pose_msg.pose.position.y = float(obs['y'])
        self.obstacle_pub.publish(self._pose_msg)

        dist = math.hypot(obs['x'], obs['y'])
        flag_msg = Int32()
        flag_msg.data = 1 if dist < self.close_threshold else 0
        self.close_pub.publish(flag_msg)

    def publish_marker(self, obs: dict, use_map: bool = True):
        """
        Publish marker in 'map' frame if current_pose is available (no TF lookup needed in RViz).
        Otherwise publish in 'base_link' as a fallback.
        """
        m = self._marker_msg
        # If caller requested map-frame publishing and we have a current_pose, place
        # the marker in the map frame. Otherwise publish in the robot frame (base_link).
        if use_map and (self.current_pose is not None):
            pose = self.current_pose
            ct = math.cos(pose['theta'])
            st = math.sin(pose['theta'])
            world_x = pose['x'] + ct * obs['x'] - st * obs['y']
            world_y = pose['y'] + st * obs['x'] + ct * obs['y']
            m.header.frame_id = 'map'
            m.pose.position.x = float(world_x)
            m.pose.position.y = float(world_y)
            m.pose.position.z = 0.0
        else:
            # Publish relative to the robot (scan-updates should use this path)
            m.header.frame_id = 'base_link'
            m.pose.position.x = float(obs['x'])
            m.pose.position.y = float(obs['y'])
            m.pose.position.z = 0.0

        # Ensure scale and color are set every time (avoid accidental zeroing)
        m.scale.x = 0.2
        m.scale.y = 0.2
        m.scale.z = 0.2
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0

        m.pose.orientation.w = 1.0
        m.action = Marker.ADD
        try:
            from rclpy.duration import Duration
            m.lifetime = Duration(seconds=1.0).to_msg()
        except Exception:
            pass
        m.header.stamp = self.get_clock().now().to_msg()
        self.marker_pub.publish(m)
        self.get_logger().debug(f'Published ADD marker at ({m.pose.position.x:.3f}, {m.pose.position.y:.3f}) frame={m.header.frame_id}')

    def _publish_delete_marker(self):
        m = self._delete_marker
        # Match frame to how markers are being published (map when we have robot pose, else base_link)
        if self.current_pose is not None:
            m.header.frame_id = 'map'
        else:
            m.header.frame_id = 'base_link'
        # header.stamp must be current so RViz processes the delete
        m.header.stamp = self.get_clock().now().to_msg()
        self.marker_pub.publish(m)
        self.get_logger().debug(f'Published DELETE marker frame={m.header.frame_id}')

    def delete_marker(self):
        self._publish_delete_marker()

    def _send_transform(self, pose: dict, stamp=None):
        """
        Publish a map -> base_link transform using the provided pose.
        If stamp is provided (a builtin_interfaces/Time), use it; otherwise use current time.
        """
        t = TransformStamped()
        if stamp is not None:
            t.header.stamp = stamp
        else:
            t.header.stamp = self.get_clock().now().to_msg()

        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x = float(pose['x'])
        t.transform.translation.y = float(pose['y'])
        t.transform.translation.z = 0.0
        theta = float(pose['theta'])
        t.transform.rotation.z = math.sin(theta / 2.0)
        t.transform.rotation.w = math.cos(theta / 2.0)
        # Send the transform (event-driven / stamped)
        self.tf_broadcaster.sendTransform(t)

    def destroy_node(self):
        # delete marker and shutdown pool
        self._publish_delete_marker()
        try:
            self.pool.shutdown(wait=False)
        except Exception:
            pass
        super().destroy_node()

    #
    # Debug image generation (with scan number and legend)
    #
    def _save_debug_image(self,
                          scan: LaserScan,
                          raw_pts: np.ndarray,
                          filtered_pts: np.ndarray,
                          accepted_clusters: List[np.ndarray],
                          rejected_clusters: List[np.ndarray],
                          pose: dict,
                          label: str = 'debug',
                          scan_num: Optional[int] = None,
                          pose_present: bool = False):
        """
        Generate and save a debug image. Filename will include the scan number (if provided).
        pose_present indicates whether the pose used is the real robot pose or a fallback.

        This function will honor either debug_level or the forced counter:
        - if debug_level == 0 and forced counter == 0 => no-op
        - otherwise it saves and, if forced counter > 0, decrements it.
        """
        # Only proceed if PIL is available and either debug level enabled or forced counter > 0
        if not PIL_AVAILABLE:
            return

        with self.lock:
            forced_remaining = int(self.forced_debug_remaining)
            dbg_enabled = (self.debug_level > 0) or (forced_remaining > 0)

        if not dbg_enabled:
            return

        # if scan_num not provided, use current counter
        if scan_num is None:
            with self.lock:
                scan_num = int(self.scan_counter)

        def robot_to_world(pts: np.ndarray, pose: dict):
            if pts.size == 0:
                return np.empty((0, 2), dtype=np.float64)
            ct = math.cos(pose['theta'])
            st = math.sin(pose['theta'])
            px = pts[:, 0].astype(np.float64)
            py = pts[:, 1].astype(np.float64)
            wx = pose['x'] + ct * px - st * py
            wy = pose['y'] + st * px + ct * py
            return np.stack((wx, wy), axis=-1)

        raw_world = robot_to_world(raw_pts, pose) if raw_pts.size else np.empty((0, 2))
        filtered_world = robot_to_world(filtered_pts, pose) if filtered_pts.size else np.empty((0, 2))
        accepted_world = [robot_to_world(c, pose) for c in accepted_clusters]
        rejected_world = [robot_to_world(c, pose) for c in rejected_clusters]

        # Map world extents (if map available)
        if self.map_info and self.map_np is not None:
            ox = self.map_info['ox']
            oy = self.map_info['oy']
            width = self.map_info['width']
            height = self.map_info['height']
            res = self.map_info['res']
            yaw = self.map_info.get('yaw', 0.0)
            # Get the world coords of the four corners of the map
            corners_map = np.array([[0, 0], [width * res, 0], [0, height * res], [width * res, height * res]])
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            world_corners = []
            for mx, my in corners_map:
                wx = ox + cos_y * mx - sin_y * my
                wy = oy + sin_y * mx + cos_y * my
                world_corners.append((wx, wy))
            world_corners = np.array(world_corners)
            map_min_x = float(world_corners[:, 0].min())
            map_max_x = float(world_corners[:, 0].max())
            map_min_y = float(world_corners[:, 1].min())
            map_max_y = float(world_corners[:, 1].max())
        else:
            # No map: define minimal box around robot
            map_min_x = pose['x'] - 0.5
            map_max_x = pose['x'] + 0.5
            map_min_y = pose['y'] - 0.5
            map_max_y = pose['y'] + 0.5
            res = 0.05  # arbitrary resolution for debug canvas

        # Combined extents: include raw and filtered points and clusters
        xs = []
        ys = []
        if raw_world.size:
            xs.append(raw_world[:, 0].min()); xs.append(raw_world[:, 0].max())
            ys.append(raw_world[:, 1].min()); ys.append(raw_world[:, 1].max())
        if filtered_world.size:
            xs.append(filtered_world[:, 0].min()); xs.append(filtered_world[:, 0].max())
            ys.append(filtered_world[:, 1].min()); ys.append(filtered_world[:, 1].max())
        for c in accepted_world + rejected_world:
            if c.size:
                xs.append(c[:, 0].min()); xs.append(c[:, 0].max())
                ys.append(c[:, 1].min()); ys.append(c[:, 1].max())

        if xs:
            pts_min_x = min(xs)
            pts_max_x = max(xs)
            pts_min_y = min(ys)
            pts_max_y = max(ys)
        else:
            pts_min_x = pts_max_x = pose['x']
            pts_min_y = pts_max_y = pose['y']

        world_min_x = min(map_min_x, pts_min_x) - self.debug_margin_m
        world_max_x = max(map_max_x, pts_max_x) + self.debug_margin_m
        world_min_y = min(map_min_y, pts_min_y) - self.debug_margin_m
        world_max_y = max(map_max_y, pts_max_y) + self.debug_margin_m

        meters_per_cell = res
        pixels_per_meter = (self.debug_px_per_cell / meters_per_cell)

        # compute image size in pixels
        raw_w = max(1e-6, (world_max_x - world_min_x))
        raw_h = max(1e-6, (world_max_y - world_min_y))
        img_w = int(math.ceil(raw_w * pixels_per_meter))
        img_h = int(math.ceil(raw_h * pixels_per_meter))

        # Respect maximum image dimension by scaling down if necessary
        if img_w > self.debug_max_dim or img_h > self.debug_max_dim:
            scale_down = max(img_w / self.debug_max_dim, img_h / self.debug_max_dim)
            pixels_per_meter = pixels_per_meter / scale_down
            img_w = int(max(200, img_w / scale_down))
            img_h = int(max(200, img_h / scale_down))

        img_w = max(200, img_w)
        img_h = max(200, img_h)

        # Create canvas using configured neutral gray background instead of white
        img = Image.new('RGBA', (img_w, img_h), self.debug_bg_color)
        draw = ImageDraw.Draw(img, 'RGBA')

        # Try to load a slightly larger/bolder font if available, fallback to default
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        except Exception:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

        def draw_text_outline(draw_obj, position, text, font_obj, fill=(0, 0, 0), outline=(255, 255, 255)):
            """
            Draw text with a thin outline by drawing the outline color multiple times around
            the main text, then the main text on top. This increases visibility on varied backgrounds.
            """
            if font_obj is None:
                draw_obj.text(position, text, fill=fill)
                return
            x, y = position
            # outline offsets (8-direction)
            offs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
            for dx, dy in offs:
                draw_obj.text((x + dx, y + dy), text, font=font_obj, fill=outline)
            draw_obj.text((x, y), text, font=font_obj, fill=fill)

        def w2p(wx, wy):
            px = (wx - world_min_x) * pixels_per_meter
            # image y coordinates go down, world y usually up -> invert
            py = img_h - (wy - world_min_y) * pixels_per_meter
            return int(px), int(py)

        # Draw semi-transparent map overlay (sample cells at stride if map large)
        if self.map_info and self.map_np is not None:
            cos_y = math.cos(self.map_info.get('yaw', 0.0))
            sin_y = math.sin(self.map_info.get('yaw', 0.0))
            ox = self.map_info['ox']
            oy = self.map_info['oy']
            width = self.map_info['width']
            height = self.map_info['height']
            res = self.map_info['res']
            stride = 1
            if width * height > 200000:
                stride = max(1, int(math.sqrt((width * height) / 200000)))
            cell_px = max(1, int(math.ceil(res * pixels_per_meter)))
            for cy in range(0, height, stride):
                for cx in range(0, width, stride):
                    val = int(self.map_np[cy, cx])
                    # map local coords
                    map_x = (cx + 0.5) * res
                    map_y = (cy + 0.5) * res
                    wx = ox + cos_y * map_x - sin_y * map_y
                    wy = oy + sin_y * map_x + cos_y * map_y
                    px, py = w2p(wx, wy)
                    half = int(math.ceil(cell_px / 2.0))
                    bbox = [px - half, py - half, px + half, py + half]
                    if val == 0:
                        color = (245, 245, 245, 220)  # free (lighter than background)
                    elif val == -1:
                        color = (210, 210, 210, 200)  # unknown
                    else:
                        color = (40, 40, 40, 230)  # occupied (dark)
                    draw.rectangle(bbox, fill=color)

        # Draw distance rings around robot (0.5m steps up to 3m)
        max_ring = 3.0
        ring_step = 0.5
        for r in np.arange(ring_step, max_ring + ring_step, ring_step):
            radius_px = r * pixels_per_meter
            center_px = w2p(pose['x'], pose['y'])
            bbox = [center_px[0] - radius_px, center_px[1] - radius_px, center_px[0] + radius_px, center_px[1] + radius_px]
            draw.ellipse(bbox, outline=(150, 150, 150, 140), width=1)

        # Draw raw scan rays as faint lines (sampled)
        if raw_world.size:
            robot_p = w2p(pose['x'], pose['y'])
            sample_stride = max(1, int(len(raw_world) / 250))
            for i in range(0, len(raw_world), sample_stride):
                p = raw_world[i]
                px, py = w2p(float(p[0]), float(p[1]))
                draw.line([robot_p, (px, py)], fill=(180, 200, 255, 60), width=1)

        # Draw raw scan points: dark blue
        for p in raw_world:
            px, py = w2p(float(p[0]), float(p[1]))
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(20, 70, 200, 220))

        # Draw filtered points: greenish
        for p in filtered_world:
            px, py = w2p(float(p[0]), float(p[1]))
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(0, 170, 80, 220))

        # Draw accepted clusters: unique color per cluster, convex hull and centroid (with count label)
        for idx, cluster in enumerate(accepted_world):
            if cluster.size == 0:
                continue
            hue = (idx * 0.618033988749895) % 1.0  # golden ratio to spread hues
            ccol = _hsv_to_rgb_tuple(hue, 0.95, 0.95, alpha=220)
            # points
            for p in cluster:
                px, py = w2p(float(p[0]), float(p[1]))
                draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=ccol)
            # convex hull (semi-transparent fill)
            try:
                hull = convex_hull(cluster)
                if len(hull) >= 3:
                    hull_px = [w2p(x, y) for x, y in hull]
                    draw.polygon(hull_px, outline=ccol, fill=(ccol[0], ccol[1], ccol[2], 80))
            except Exception:
                pass
            # centroid and label (cluster size)
            centroid = np.mean(cluster, axis=0)
            cpx, cpy = w2p(float(centroid[0]), float(centroid[1]))
            draw.ellipse([cpx - 6, cpy - 6, cpx + 6, cpy + 6], outline=(255, 255, 255, 220), width=2)
            draw_text_outline(draw, (cpx + 8, cpy - 8), f'n={cluster.shape[0]}', font, fill=(0, 0, 0), outline=(255, 255, 255))

        # Draw rejected clusters: orange points and dashed hull
        for cluster in rejected_world:
            if cluster.size == 0:
                continue
            for p in cluster:
                px, py = w2p(float(p[0]), float(p[1]))
                draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(255, 120, 20, 220))
            hull = convex_hull(cluster)
            if len(hull) >= 2:
                hull_px = [w2p(x, y) for x, y in hull]
                hull_px.append(hull_px[0])
                for i in range(len(hull_px) - 1):
                    (x1, y1) = hull_px[i]
                    (x2, y2) = hull_px[i + 1]
                    seg_len = math.hypot(x2 - x1, y2 - y1)
                    if seg_len < 4:
                        draw.line([(x1, y1), (x2, y2)], fill=(255, 140, 30, 220), width=2)
                    else:
                        nseg = max(1, int(seg_len / 8))
                        for j in range(nseg):
                            t0 = j / nseg
                            t1 = (j + 0.5) / nseg
                            sx = int(x1 + (x2 - x1) * t0)
                            sy = int(y1 + (y2 - y1) * t0)
                            ex = int(x1 + (x2 - x1) * t1)
                            ey = int(y1 + (y2 - y1) * t1)
                            draw.line([(sx, sy), (ex, ey)], fill=(255, 140, 30, 220), width=2)

        # Draw robot as a filled triangle (magenta) with heading arrow
        p_front = (pose['x'] + 0.25 * math.cos(pose['theta']), pose['y'] + 0.25 * math.sin(pose['theta']))
        p_left = (pose['x'] + 0.12 * math.cos(pose['theta'] + 2.4), pose['y'] + 0.12 * math.sin(pose['theta'] + 2.4))
        p_right = (pose['x'] + 0.12 * math.cos(pose['theta'] - 2.4), pose['y'] + 0.12 * math.sin(pose['theta'] - 2.4))
        tri = [w2p(*p_front), w2p(*p_left), w2p(*p_right)]
        draw.polygon(tri, fill=(200, 0, 200, 230))
        rcx, rcy = w2p(pose['x'], pose['y'])
        draw.ellipse([rcx - 4, rcy - 4, rcx + 4, rcy + 4], fill=(255, 255, 255, 255), outline=(120, 0, 120, 255))

        # Draw tracked obstacle (if exists) with a bright ring and connecting line
        with self.lock:
            tracked = None if self.tracked_obstacle is None else self.tracked_obstacle.copy()
        if tracked is not None:
            tw = robot_to_world(np.array([[tracked['x'], tracked['y']]], dtype=np.float64), pose)[0]
            tpx, tpy = w2p(float(tw[0]), float(tw[1]))
            draw.ellipse([tpx - 10, tpy - 10, tpx + 10, tpy + 10], outline=(255, 0, 0, 255), width=3)
            draw.line([(rcx, rcy), (tpx, tpy)], fill=(255, 0, 0, 200), width=2)
            draw_text_outline(draw, (tpx + 6, tpy - 6), 'tracked', font, fill=(255, 0, 0), outline=(255, 255, 255))

        # Legend box (explicit color/key mapping). Use more diverse palette and outlined text for readability.
        legend_x = 8
        legend_y = img_h - 160
        lw = 420
        lh = 152
        draw.rectangle([legend_x - 6, legend_y - 6, legend_x + lw, legend_y + lh], fill=(255, 255, 255, 230), outline=(80, 80, 80, 220))
        ly = legend_y + 10

        # 1) Raw points (blue)
        draw.rectangle([legend_x + 10, ly, legend_x + 34, ly + 16], fill=(20, 70, 200, 220))
        draw_text_outline(draw, (legend_x + 38, ly), 'raw scan points (blue)', font, fill=(0, 0, 0), outline=(255, 255, 255))
        ly += 22

        # 2) Filtered points (bright lime)
        draw.rectangle([legend_x + 10, ly, legend_x + 34, ly + 16], fill=(0, 220, 80, 240))
        draw_text_outline(draw, (legend_x + 38, ly), 'filtered points (free space, lime)', font, fill=(0, 0, 0), outline=(255, 255, 255))
        ly += 22

        # 3) Accepted clusters palette: draw several highly distinct colors
        palette = [
            (220, 20, 60, 220),   # crimson
            (30, 144, 255, 220),  # dodger blue
            (50, 205, 50, 220),   # lime green
            (255, 140, 0, 220),   # dark orange
            (138, 43, 226, 220),  # blueviolet
            (0, 206, 209, 220),   # darkturquoise
        ]
        # draw small boxes for palette
        px0 = legend_x + 10
        for i, col in enumerate(palette):
            x0 = px0 + i * 20
            draw.rectangle([x0, ly, x0 + 16, ly + 16], fill=col)
        draw_text_outline(draw, (px0 + len(palette) * 20 + 6, ly), 'accepted clusters (distinct colors)', font, fill=(0, 0, 0), outline=(255, 255, 255))
        ly += 22

        # 4) Rejected clusters (orange)
        draw.rectangle([legend_x + 10, ly, legend_x + 34, ly + 16], fill=(255, 120, 20, 220))
        draw_text_outline(draw, (legend_x + 38, ly), 'rejected clusters (too large) - orange', font, fill=(0, 0, 0), outline=(255, 255, 255))
        ly += 22

        # 5) Robot & tracked
        draw.polygon([ (legend_x + 18, ly + 6), (legend_x + 10, ly + 18), (legend_x + 26, ly + 18) ], fill=(200, 0, 200, 230))
        draw_text_outline(draw, (legend_x + 38, ly), 'robot pose (magenta triangle)', font, fill=(0, 0, 0), outline=(255, 255, 255))
        ly += 22
        draw.rectangle([legend_x + 10, ly, legend_x + 34, ly + 16], fill=(255, 0, 0, 255))
        draw_text_outline(draw, (legend_x + 38, ly), 'tracked obstacle (red ring + label)', font, fill=(0, 0, 0), outline=(255, 255, 255))

        # Scale bar (meters)
        sb_len_m = 1.0
        sb_len_px = int(sb_len_m * pixels_per_meter)
        sb_x = img_w - 160
        sb_y = img_h - 28
        draw.line([(sb_x, sb_y), (sb_x + sb_len_px, sb_y)], fill=(0, 0, 0), width=3)
        draw.line([(sb_x, sb_y - 6), (sb_x, sb_y + 6)], fill=(0, 0, 0), width=2)
        draw.line([(sb_x + sb_len_px, sb_y - 6), (sb_x + sb_len_px, sb_y + 6)], fill=(0, 0, 0), width=2)
        draw_text_outline(draw, (sb_x, sb_y - 18), f'{sb_len_m} m', font, fill=(0, 0, 0), outline=(255, 255, 255))

        # Text overlay with metadata (use outlined text to improve visibility)
        ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        info_lines = [
            f'label: {label}',
            f'ts: {ts}',
            f'scan_num: {scan_num}',
            f'raw_pts: {len(raw_pts) if raw_pts is not None else 0}',
            f'filtered_pts: {len(filtered_pts) if filtered_pts is not None else 0}',
            f'accepted_clusters: {len(accepted_clusters)}',
            f'rejected_clusters: {len(rejected_clusters)}',
            f'pose_present: {pose_present}',
            f'pose: x={pose["x"]:.2f} y={pose["y"]:.2f} th={pose["theta"]:.2f}',
            f'min_point_distance: {self.min_point_distance:.2f} m',
        ]
        tx = 8
        ty = 8
        for line in info_lines:
            draw_text_outline(draw, (tx, ty), line, font, fill=(0, 0, 0), outline=(255, 255, 255))
            ty += 16

        # Save image with scan number in filename
        filename = f'scan{scan_num:06d}_{label}_{datetime.utcnow().strftime("%Y%m%dT%H%M%S.%f")}.png'
        outpath = self.debug_folder / filename
        try:
            img.save(str(outpath))
            self.get_logger().info(f'Saved debug image: {outpath} (size: {img_w}x{img_h})')
            # If this save was due to forced counter, decrement it
            with self.lock:
                if self.forced_debug_remaining > 0:
                    self.forced_debug_remaining -= 1
                    self.get_logger().info(f'Forced debug images remaining: {self.forced_debug_remaining}')
        except Exception as e:
            self.get_logger().warning(f'Failed to save debug image to {outpath}: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()