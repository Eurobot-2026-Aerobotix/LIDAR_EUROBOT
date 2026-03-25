#!/usr/bin/env python3
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3, TransformStamped, Twist
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster
from rclpy.node import Node
from std_msgs.msg import Int32
from lidar_obstacle_warning_msgs.msg import EmergencyStop

# QoS imports for transient-local map subscription
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy


class ObstacleTracker(Node):
    """
    Obstacle detection node for clustering LIDAR scans and publishing obstacles.
    Includes collision prediction with trajectory forecasting.
    """

    def __init__(self):
        super().__init__('obstacle_tracker')

        # ====== COLLISION PREDICTION PARAMETERS ======
        # Robot collision geometry (rectangle)
        self.robot_length = 0.5  # meters
        self.robot_width = 0.3   # meters
        
        # Obstacle collision geometry (circle)
        self.obstacle_collision_radius = 0.25  # meters
        
        # Detection threshold - start prediction when obstacle is within this distance
        self.detection_threshold = 3  # meters (increased from 1.0 for better reaction time)
        
        # Collision prediction time horizon (accounts for system latency)
        # Total prediction time = fixed_offset
        #                       + robot_velocity_factor * robot_speed
        #                       + obstacle_velocity_factor * obstacle_speed
        # Observed latency:  0.1m at 0.7m/s = ~0.143s
        self.collision_prediction_time_fixed = 0.16  # seconds - base latency compensation
        self.collision_prediction_time_velocity_factor = 0.40  # seconds per m/s - robot velocity-dependent offset
        self.collision_prediction_time_obstacle_velocity_factor = 0.35  # seconds per m/s - obstacle velocity-dependent offset
        
        # Number of intermediate steps to check along trajectory
        self.prediction_steps = 5  # Check 5 intermediate positions (increased from 3)
        
        # History length for velocity estimation
        self.history_length = 10
        
        # Low-pass filter for obstacle position (exponential moving average)
        # alpha = 1.0 means no filtering, alpha = 0.1 means heavy smoothing
        self.obstacle_filter_alpha = 0.4  # Smoothing factor (tune based on noise level)
        self.filtered_obstacle_map = None  # Stores filtered position in MAP frame (rotation-invariant)
        
        # Rotation detection parameters - use moderate filtering during rotation
        self.rotation_rate_threshold = 0.5  # rad/s - increased from 0.3 to only trigger on significant rotation
        self.rotation_filter_alpha = 0.25  # Moderate smoothing during rotation (was 0.05 which was too heavy)
        self.rotation_damping_alpha = self.rotation_filter_alpha

        # Safety margin tuning for LD19 @ 10Hz and fast relative motion
        self.scan_rate_hz = 10.0
        self.emergency_reaction_time = 0.07  # seconds
        self.base_safety_margin = 0.03       # meters
        self.max_safety_margin = 0.60        # meters
        self.reaction_time_scale = 0.40      # less conservative for static scenes
        self.static_obstacle_speed_threshold = 0.20  # m/s
        self.static_dynamic_margin_scale = 0.20  # Scale down margin for static obstacles to avoid over-triggering

        # Supervisor-based safety logic (recommended primary stop criterion)
        self.use_supervisor_logic = True
        self.use_dynamic_radius_for_geometry = False
        self.robot_collision_radius = 0.5 * math.hypot(self.robot_length, self.robot_width)
        self.supervisor_brake_deceleration = 1.3   # m/s^2 conservative effective braking
        self.supervisor_total_latency = 0.24       # s sensing + compute + command + actuation
        self.supervisor_buffer = 0.07              # m model and estimation uncertainty
        self.supervisor_ttc_critical = 0.70        # s emergency trigger threshold
        self.supervisor_ttc_release = 0.85         # s hysteresis release threshold
        self.supervisor_release_margin = 0.10      # m hysteresis on safe clearance
        self.supervisor_min_hold_time = 0.20       # s minimum hold after trigger
        self.supervisor_geometry_ttc_gate = 0.40   # s: allow geometry to trigger only for near-term risk
        self.supervisor_geometry_clearance_gate = 0.04  # m: allow geometry if very close to required clearance
        self.supervisor_max_robot_speed = 1.4       # m/s clamp for noisy speed estimates
        self.supervisor_max_closing_speed = 2.5     # m/s clamp for noisy relative speed estimates
        self.supervisor_static_closing_threshold = 0.20  # m/s
        self.supervisor_static_clearance_relax = 0.04    # m reduce required clearance in near-static approach
        self.supervisor_emergency_active = False
        self.supervisor_last_trigger_time = 0.0
        self.supervisor_debug_logs = True
        self.supervisor_debug_log_period = 0.5  # seconds
        self.supervisor_last_debug_log_time = 0.0
        
        # Velocity histories
        self.robot_pose_history = deque(maxlen=self.history_length)
        self.obstacle_position_history = deque(maxlen=self.history_length)
        
        # ============================================

        # Original parameters
        self.scan_skip = 1
        self.scan_skip = 1
        self.range_downsample = 1
        self.cluster_min_points = 5
        self.cluster_min_distance = 0.1
        self.cluster_max_diameter = 0.3
        self.close_threshold = 0.5
        self.scan_tracking_stop_distance = 0.5  # Increased from 0.8m to maintain tracking lock
        self.min_point_distance = 0.2

        # Subscriptions / publishers
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        qos_map = QoSProfile(depth=1)
        qos_map.reliability = ReliabilityPolicy.RELIABLE
        qos_map.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos_map.history = HistoryPolicy.KEEP_LAST
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_map)

        self.create_subscription(Vector3, '/robot_pose_raw', self.pose_callback, 10)
        self.obstacle_pub = self.create_publisher(PoseStamped, '/obstacle_relative', 10)
        self.obstacle_speed_pub = self.create_publisher(Twist, '/obstacle_relative_speed', 10)
        self.close_pub = self.create_publisher(Int32, '/emergency_break', 10)
        self.emergency_stop_pub = self.create_publisher(EmergencyStop, '/emergency_stop', 10)
        self.marker_pub = self.create_publisher(Marker, '/obstacle_candidate_marker', 10)
        
        # New publishers for collision visualization
        self.collision_viz_pub = self.create_publisher(Marker, '/collision_visualization', 10)
        self.collision_timer = self.create_timer(0.05, self.collision_check_callback)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Cached state
        self.latest_scan:  Optional[LaserScan] = None
        self.latest_map_msg: Optional[OccupancyGrid] = None
        self.map_np:  Optional[np.ndarray] = None
        self.map_info = {}
        self.current_pose = None
        self.tracked_obstacle = None  # Now stored in MAP frame (not robot frame)
        self.tracking = False

        # Helpers
        self.scan_counter = 0
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=2)

        # Reused message objects
        self._pose_msg = PoseStamped()
        self._twist_msg = Twist()
        self._marker_msg = Marker()
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

        self._delete_marker = Marker()
        self._delete_marker.ns = 'obstacle'
        self._delete_marker.id = 0
        self._delete_marker.action = Marker.DELETE
        self._delete_marker.type = Marker.SPHERE

        self.get_logger().info(
            f'ObstacleTracker starting:\n'
            f'  Robot size: {self.robot_length}m × {self.robot_width}m\n'
            f'  Obstacle radius: {self.obstacle_collision_radius}m\n'
            f'  Detection threshold: {self.detection_threshold}m\n'
            f'  Prediction time fixed: {self.collision_prediction_time_fixed}s\n'
            f'  Prediction time robot velocity factor: {self.collision_prediction_time_velocity_factor}s per m/s\n'
            f'  Prediction time obstacle velocity factor: {self.collision_prediction_time_obstacle_velocity_factor}s per m/s\n'
            f'  Prediction steps: {self.prediction_steps}\n'
            f'  Obstacle filter alpha: {self.obstacle_filter_alpha}\n'
            f'  Rotation rate threshold: {self.rotation_rate_threshold} rad/s\n'
            f'  Rotation filter alpha: {self.rotation_filter_alpha} (moderate filtering, not heavy damping)\n'
            f'  Scan rate: {self.scan_rate_hz}Hz, reaction time: {self.emergency_reaction_time}s\n'
            f'  Safety margin: base={self.base_safety_margin}m, max={self.max_safety_margin}m\n'
            f'  Supervisor: enabled={self.use_supervisor_logic}, a_brake={self.supervisor_brake_deceleration}m/s^2, '
            f'latency={self.supervisor_total_latency}s, buffer={self.supervisor_buffer}m, '
            f'ttc_crit={self.supervisor_ttc_critical}s, geom_ttc_gate={self.supervisor_geometry_ttc_gate}s\n'
            f'  Supervisor debug logs: {self.supervisor_debug_logs}, period={self.supervisor_debug_log_period}s'
        )

    #
    # Callbacks
    #
    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        with self.lock:
            self.scan_counter += 1
            current_scan_num = int(self.scan_counter)

        # Get scan timestamp for pose interpolation
        scan_timestamp = None
        try:
            scan_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        except Exception:
            scan_timestamp = self.get_clock().now().nanoseconds / 1e9

        # Interpolate pose at scan timestamp for accurate transform
        pose_for_tf = self.interpolate_pose(scan_timestamp)
        if pose_for_tf is None:
            with self.lock:
                pose_for_tf = None if self.current_pose is None else self.current_pose.copy()

        if pose_for_tf is not None: 
            stamp = None
            try:
                stamp = msg.header.stamp
            except Exception:
                stamp = None
            self._send_transform(pose_for_tf, stamp=stamp)

        if (current_scan_num - 1) % self.scan_skip != 0:
            return

        try:
            self.pool.submit(self._process_scan, msg, current_scan_num)
        except Exception: 
            self._process_scan(msg, current_scan_num)

    def _process_scan(self, scan: LaserScan, scan_num: int):
        """
        Main scan processing - handles both detection and tracking.
        Uses interpolated pose at scan timestamp for accuracy.
        """
        # Get interpolated pose at scan timestamp
        scan_timestamp = None
        try:
            scan_timestamp = scan.header.stamp.sec + scan.header.stamp.nanosec / 1e9
        except Exception:
            scan_timestamp = self.get_clock().now().nanoseconds / 1e9
        
        interpolated_pose = self.interpolate_pose(scan_timestamp)
        if interpolated_pose is None:
            with self.lock:
                if self.current_pose is None:
                    return  # No pose available yet
                interpolated_pose = self.current_pose.copy()

        # Check if we have map data for detection
        has_map = self.latest_map_msg is not None and self.map_np is not None

        with self.lock:
            tracking = self.tracking
            tracked_exists = self.tracked_obstacle is not None

        if not tracking or not tracked_exists:
            # Not tracking - try to detect new obstacle (requires map)
            if has_map:
                if self._detect_new_obstacle(scan, interpolated_pose):
                    with self.lock:
                        obs = self.tracked_obstacle.copy() if self.tracked_obstacle else None
                    if obs:
                        self.publish_obstacle(obs)
                        self.publish_marker(obs)
                else:
                    self._publish_delete_marker()
            return

        # Currently tracking - update obstacle position
        pts, _ = self.scan_to_points(scan, downsample=self.range_downsample, return_indices=True)
        if pts.size == 0:
            return

        all_clusters, _ = self.cluster_scan_ordered(
            pts,
            min_points=self.cluster_min_points,
            max_gap=self.cluster_min_distance,
            max_diameter=self.cluster_max_diameter,
        )

        # Filter clusters to only keep those in free space (not in black zones)
        accepted_clusters = []
        if has_map:
            for c in all_clusters:
                filtered = self.filter_free_space(c, interpolated_pose)
                # Only accept cluster if ALL points pass the filter (are in free space)
                if filtered.shape[0] == c.shape[0] and filtered.shape[0] >= self.cluster_min_points:
                    accepted_clusters.append(c)
        else:
            accepted_clusters = all_clusters

        if not accepted_clusters:
            # No clusters found - try re-detection if we have map
            if has_map:
                with self.lock:
                    self.tracking = False
                    self.tracked_obstacle = None
                    self.filtered_obstacle_map = None
                if self._detect_new_obstacle(scan, interpolated_pose):
                    with self.lock:
                        obs = self.tracked_obstacle.copy() if self.tracked_obstacle else None
                    if obs:
                        self.publish_obstacle(obs)
                        self.publish_marker(obs)
                else:
                    self._publish_delete_marker()
            return

        # Convert tracked obstacle from map frame to robot frame for cluster matching
        with self.lock:
            obs_map = self.tracked_obstacle.copy()
        ct = math.cos(interpolated_pose['theta'])
        st = math.sin(interpolated_pose['theta'])
        # Map to robot frame: robot = R(-theta) * (map - pose)
        dx = obs_map['x'] - interpolated_pose['x']
        dy = obs_map['y'] - interpolated_pose['y']
        prev_robot_x = ct * dx + st * dy   # cos(-theta)*dx - sin(-theta)*dy = cos(theta)*dx + sin(theta)*dy
        prev_robot_y = -st * dx + ct * dy  # sin(-theta)*dx + cos(-theta)*dy = -sin(theta)*dx + cos(theta)*dy
        prev = np.array([prev_robot_x, prev_robot_y])
        
        centroids = [np.mean(c, axis=0) for c in accepted_clusters]
        dists = [np.linalg.norm(c - prev) for c in centroids]
        best_idx = int(np.argmin(dists))
        best_centroid = centroids[best_idx]
        best_dist = dists[best_idx]

        if best_dist > self.scan_tracking_stop_distance:
            # Lost tracking - try re-detection if we have map
            if has_map:
                with self.lock:
                    self.tracking = False
                    self.tracked_obstacle = None
                    self.filtered_obstacle_map = None
                if self._detect_new_obstacle(scan, interpolated_pose):
                    with self.lock:
                        obs = self.tracked_obstacle.copy() if self.tracked_obstacle else None
                    if obs:
                        self.publish_obstacle(obs)
                        self.publish_marker(obs)
                else:
                    self._publish_delete_marker()
            else:
                with self.lock:
                    self.tracking = False
                    self.tracked_obstacle = None
                    self.filtered_obstacle_map = None
                self._publish_delete_marker()
            return

        # Update tracked obstacle with filtering in MAP FRAME
        # This ensures robot rotation doesn't cause position errors
        
        # Detect if robot is currently rotating - if so, heavily dampen updates
        # to prevent false obstacle movement due to pose interpolation errors
        robot_vel = self.estimate_robot_velocity()
        is_rotating = abs(robot_vel['vtheta']) > self.rotation_rate_threshold
        
        raw_x = float(best_centroid[0])
        raw_y = float(best_centroid[1])
        
        # Convert raw measurement from robot frame to map frame
        raw_map_x = interpolated_pose['x'] + ct * raw_x - st * raw_y
        raw_map_y = interpolated_pose['y'] + st * raw_x + ct * raw_y
        
        # Validate that the new position is in a valid map area
        position_valid = self.is_valid_map_position(raw_map_x, raw_map_y)
        
        with self.lock:
            if not position_valid:
                # Position is in invalid area (outside map or occupied)
                # During rotation this is expected - just skip this update instead of resetting
                if is_rotating and self.tracked_obstacle is not None:
                    # Keep existing position, don't update with invalid measurement
                    obs = self.tracked_obstacle.copy()
                    # Skip to publishing - don't reset tracking
                else:
                    # Not rotating, genuinely invalid - reset tracking
                    self.tracking = False
                    self.tracked_obstacle = None
                    self.filtered_obstacle_map = None
                    obs = None
            else:
                # Position is valid - apply low-pass filter in map frame (rotation-invariant)
                # Use much lower alpha during rotation to prevent false drift
                if is_rotating:
                    alpha = self.rotation_damping_alpha
                else:
                    alpha = self.obstacle_filter_alpha
                    
                if self.filtered_obstacle_map is None:
                    self.filtered_obstacle_map = {'x': raw_map_x, 'y': raw_map_y}
                else:
                    self.filtered_obstacle_map['x'] = alpha * raw_map_x + (1 - alpha) * self.filtered_obstacle_map['x']
                    self.filtered_obstacle_map['y'] = alpha * raw_map_y + (1 - alpha) * self.filtered_obstacle_map['y']
                
                # Validate filtered position is still valid
                if not self.is_valid_map_position(self.filtered_obstacle_map['x'], self.filtered_obstacle_map['y']):
                    if is_rotating:
                        # During rotation, revert to pre-filter position
                        self.filtered_obstacle_map = {'x': self.tracked_obstacle['x'], 'y': self.tracked_obstacle['y']}
                        obs = self.tracked_obstacle.copy()
                    else:
                        self.tracking = False
                        self.tracked_obstacle = None
                        self.filtered_obstacle_map = None
                        obs = None
                else:
                    # Store tracked_obstacle in MAP FRAME
                    self.tracked_obstacle = {'x': self.filtered_obstacle_map['x'], 'y': self.filtered_obstacle_map['y']}
                    
                    timestamp = self.get_clock().now().nanoseconds / 1e9
                    # Store map-frame position in history (no pose needed for conversion)
                    self.obstacle_position_history.append({
                        'x': self.tracked_obstacle['x'],
                        'y': self.tracked_obstacle['y'],
                        'timestamp': timestamp
                    })
                    obs = self.tracked_obstacle.copy()
        
        # If tracking was reset, try re-detection
        with self.lock:
            still_tracking = self.tracking
        
        if not still_tracking:
            if has_map:
                if self._detect_new_obstacle(scan, interpolated_pose):
                    with self.lock:
                        obs = self.tracked_obstacle.copy() if self.tracked_obstacle else None
                    if obs:
                        self.publish_obstacle(obs)
                        self.publish_marker(obs)
                else:
                    self._publish_delete_marker()
            return

        self.publish_obstacle(obs)
        self.publish_marker(obs)

    def map_callback(self, msg: OccupancyGrid):
        self.latest_map_msg = msg
        info = msg.info
        width = info.width
        height = info.height
        if isinstance(msg.data, (bytes, bytearray)):
            arr = np.frombuffer(msg.data, dtype=np.int8)
        else:
            arr = np.array(msg.data, dtype=np.int8)
        try:
            arr = arr.reshape((height, width))
        except Exception:
            arr = np.array(msg.data, dtype=np.int8).reshape((height, width))
        self.map_np = arr

        ox = info.origin.position.x
        oy = info.origin.position.y
        q = info.origin.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.map_info = {
            'res': info.resolution,
            'width': width,
            'height':  height,
            'ox': ox,
            'oy': oy,
            'yaw': yaw,
        }

    def pose_callback(self, msg: Vector3):
        """Only updates pose and history - scan_callback handles all processing."""
        pose = {'x': float(msg.x), 'y': float(msg.y), 'theta': float(msg.z)}
        timestamp = self.get_clock().now().nanoseconds / 1e9
        
        with self.lock:
            self.current_pose = pose
            self.robot_pose_history.append({
                'x': pose['x'],
                'y': pose['y'],
                'theta': pose['theta'],
                'timestamp': timestamp
            })

    def _detect_new_obstacle(self, scan: LaserScan, pose: dict) -> bool:
        """
        Attempt to detect a new obstacle from scratch using map filtering.
        Returns True if a new obstacle was detected and tracking started.
        """
        pts, idxs = self.scan_to_points(scan, downsample=self.range_downsample, return_indices=True)
        if pts.size == 0:
            return False

        clusters, _ = self.cluster_scan_ordered(
            pts,
            min_points=self.cluster_min_points,
            max_gap=self.cluster_min_distance,
            max_diameter=None,
        )

        accepted_clusters: List[np.ndarray] = []
        for c in clusters:
            filtered = self.filter_free_space(c, pose)
            if filtered.shape[0] != c.shape[0]:
                continue
            if filtered.shape[0] < self.cluster_min_points:
                continue
            if self.cluster_max_diameter is not None:
                if filtered.shape[0] <= 1:
                    diam = 0.0
                else:
                    diffs = filtered[:, None, :] - filtered[None, :, :]
                    dists = np.hypot(diffs[..., 0], diffs[..., 1])
                    diam = float(dists.max())
                if diam > self.cluster_max_diameter:
                    continue
            accepted_clusters.append(filtered)

        if not accepted_clusters:
            return False

        # Sort by size (largest first), but validate each candidate's map position
        sorted_clusters = sorted(accepted_clusters, key=lambda c: c.shape[0], reverse=True)
        
        ct = math.cos(pose['theta'])
        st = math.sin(pose['theta'])
        
        for cluster in sorted_clusters:
            centroid = np.mean(cluster, axis=0)
            raw_x = float(centroid[0])
            raw_y = float(centroid[1])
            
            # Convert to map frame
            map_x = pose['x'] + ct * raw_x - st * raw_y
            map_y = pose['y'] + st * raw_x + ct * raw_y
            
            # Validate centroid is in valid map area
            if not self.is_valid_map_position(map_x, map_y):
                continue  # Skip this cluster, try next largest
            
            # Found a valid obstacle
            with self.lock:
                self.filtered_obstacle_map = {'x': map_x, 'y': map_y}
                self.tracked_obstacle = {'x': map_x, 'y': map_y}  # Stored in MAP frame
                self.tracking = True
                
                # Clear old history and start fresh for the new obstacle
                self.obstacle_position_history.clear()
                
                # Clear old history and start fresh for the new obstacle
                self.obstacle_position_history.clear()
                
                timestamp = self.get_clock().now().nanoseconds / 1e9
                # Store map-frame position in history (no pose needed)
                self.obstacle_position_history.append({
                    'x': self.tracked_obstacle['x'],
                    'y': self.tracked_obstacle['y'],
                    'timestamp': timestamp
                })
            return True
        
        # No valid obstacle found
        return False

    #
    # Pose Interpolation for Timestamp Synchronization
    #
    def interpolate_pose(self, target_timestamp: float) -> Optional[dict]:
        """
        Interpolate robot pose at a given timestamp using linear interpolation.
        
        Args:
            target_timestamp: The timestamp (in seconds) to interpolate the pose at
            
        Returns:
            Interpolated pose dict with 'x', 'y', 'theta', 'timestamp' or None if not possible
        """
        with self.lock:
            if len(self.robot_pose_history) < 2:
                # Not enough history - return current pose if available
                if len(self.robot_pose_history) == 1:
                    return dict(self.robot_pose_history[0])
                return None
            history = list(self.robot_pose_history)
        
        # Find bracketing poses
        pose_before = None
        pose_after = None
        
        for i in range(len(history) - 1):
            if history[i]['timestamp'] <= target_timestamp <= history[i + 1]['timestamp']:
                pose_before = history[i]
                pose_after = history[i + 1]
                break
        
        # Handle edge cases: target is before or after all history
        if pose_before is None:
            if target_timestamp < history[0]['timestamp']:
                # Target is before history - extrapolate backwards (limited)
                pose_before = history[0]
                pose_after = history[1] if len(history) > 1 else history[0]
            else:
                # Target is after history - extrapolate forwards (limited)
                pose_before = history[-2] if len(history) > 1 else history[-1]
                pose_after = history[-1]
        
        # Calculate interpolation factor
        dt = pose_after['timestamp'] - pose_before['timestamp']
        if dt < 1e-9:
            return dict(pose_before)
        
        t = (target_timestamp - pose_before['timestamp']) / dt
        # Clamp extrapolation to avoid wild estimates
        t = max(-0.5, min(1.5, t))
        
        # Linear interpolation for x, y
        x = pose_before['x'] + t * (pose_after['x'] - pose_before['x'])
        y = pose_before['y'] + t * (pose_after['y'] - pose_before['y'])
        
        # Interpolate theta with proper angle wrapping
        theta_diff = pose_after['theta'] - pose_before['theta']
        theta_diff = math.atan2(math.sin(theta_diff), math.cos(theta_diff))
        theta = pose_before['theta'] + t * theta_diff
        theta = math.atan2(math.sin(theta), math.cos(theta))
        
        return {
            'x': x,
            'y': y,
            'theta': theta,
            'timestamp': target_timestamp
        }

    #
    # Collision Prediction Methods
    #
    def get_collision_prediction_time(self):
        """
        Calculate dynamic collision prediction time based on robot velocity.
        
        Returns:
            float: Total prediction time = fixed_offset
                   + robot_velocity_factor * robot_speed
                   + obstacle_velocity_factor * obstacle_speed
        """
        robot_vel = self.estimate_robot_velocity()
        obstacle_vel = self.estimate_obstacle_velocity()
        robot_speed_raw = math.hypot(robot_vel['vx'], robot_vel['vy'])
        robot_speed = min(robot_speed_raw, self.supervisor_max_robot_speed)
        obstacle_speed = math.hypot(obstacle_vel['vx'], obstacle_vel['vy'])
        return (
            self.collision_prediction_time_fixed
            + self.collision_prediction_time_velocity_factor * robot_speed
            + self.collision_prediction_time_obstacle_velocity_factor * obstacle_speed
        )

    def estimate_robot_velocity(self):
        """Estimate robot velocity from pose history."""
        with self.lock:
            if len(self.robot_pose_history) < 2:
                return {'vx': 0.0, 'vy': 0.0, 'vtheta': 0.0}
            history = list(self.robot_pose_history)
        
        dt = history[-1]['timestamp'] - history[0]['timestamp']
        if dt < 1e-6:
            return {'vx': 0.0, 'vy': 0.0, 'vtheta': 0.0}
        
        dx = history[-1]['x'] - history[0]['x']
        dy = history[-1]['y'] - history[0]['y']
        dtheta = history[-1]['theta'] - history[0]['theta']
        dtheta = math.atan2(math.sin(dtheta), math.cos(dtheta))
        
        return {
            'vx': dx / dt,
            'vy': dy / dt,
            'vtheta': dtheta / dt
        }

    def estimate_obstacle_velocity(self):
        """
        Estimate obstacle velocity from position history.
        
        Since tracked_obstacle is now stored in MAP frame, we can directly
        compute velocity from the history without coordinate transformations.
        """
        with self.lock:
            if len(self.obstacle_position_history) < 2:
                return {'vx': 0.0, 'vy': 0.0}
            history = list(self.obstacle_position_history)
        
        # Positions are already in map frame - direct velocity computation
        dt = history[-1]['timestamp'] - history[0]['timestamp']
        if dt < 1e-6:
            return {'vx': 0.0, 'vy': 0.0}
        
        dx = history[-1]['x'] - history[0]['x']
        dy = history[-1]['y'] - history[0]['y']
        
        return {'vx': dx / dt, 'vy': dy / dt}

    def predict_robot_position(self, time_ahead):
        """Predict robot position after time_ahead seconds."""
        with self.lock:
            if self.current_pose is None:
                return None
            pose = self.current_pose.copy()
        
        vel = self.estimate_robot_velocity()
        
        # Simple linear prediction
        predicted_x = pose['x'] + vel['vx'] * time_ahead
        predicted_y = pose['y'] + vel['vy'] * time_ahead
        predicted_theta = pose['theta'] + vel['vtheta'] * time_ahead
        
        return {
            'x': predicted_x,
            'y': predicted_y,
            'theta': predicted_theta
        }

    def predict_obstacle_position(self, time_ahead):
        """Predict obstacle position after time_ahead seconds (in map frame)."""
        with self.lock:
            if self.tracked_obstacle is None:
                return None
            obs = self.tracked_obstacle.copy()  # Already in map frame
        
        vel = self.estimate_obstacle_velocity()
        
        # tracked_obstacle is already in map frame - no conversion needed
        predicted_x = obs['x'] + vel['vx'] * time_ahead
        predicted_y = obs['y'] + vel['vy'] * time_ahead
        
        return {
            'x': predicted_x,
            'y': predicted_y
        }

    def check_rectangle_circle_collision(self, robot_pos, obstacle_pos, collision_radius: Optional[float] = None):
        """
        Check collision between robot rectangle and obstacle circle.
        
        Args:
            robot_pos: dict with 'x', 'y', 'theta' (map frame)
            obstacle_pos:  dict with 'x', 'y' (map frame)
        
        Returns:
            bool:  True if collision detected
        """
        if robot_pos is None or obstacle_pos is None:
            return False
        
        # Robot rectangle corners (centered at robot position)
        half_length = self.robot_length / 2.0
        half_width = self.robot_width / 2.0
        
        # Rectangle corners in robot-local frame
        corners_local = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ]
        
        # Transform corners to map frame
        cos_theta = math.cos(robot_pos['theta'])
        sin_theta = math.sin(robot_pos['theta'])
        corners_map = []
        for lx, ly in corners_local: 
            mx = robot_pos['x'] + cos_theta * lx - sin_theta * ly
            my = robot_pos['y'] + sin_theta * lx + cos_theta * ly
            corners_map.append((mx, my))
        
        # Circle center
        cx = obstacle_pos['x']
        cy = obstacle_pos['y']
        radius = self.obstacle_collision_radius if collision_radius is None else float(collision_radius)
        
        # Test 1: Is circle center inside rectangle?
        if self._point_in_rectangle(cx, cy, corners_map):
            return True
        
        # Test 2: Is any corner inside the circle?
        for corner_x, corner_y in corners_map:
            dist = math.hypot(corner_x - cx, corner_y - cy)
            if dist <= radius:
                return True
        
        # Test 3: Does circle intersect any edge?
        for i in range(4):
            p1 = corners_map[i]
            p2 = corners_map[(i + 1) % 4]
            if self._circle_intersects_segment(cx, cy, radius, p1, p2):
                return True
        
        return False

    def get_dynamic_collision_radius(self) -> float:
        """
        Dynamic collision radius to compensate for sensing and stopping delay.
        Uses line-of-sight closing speed (not total speed) to avoid
        over-triggering for static/lateral obstacles.
        """
        robot_vel = self.estimate_robot_velocity()
        obstacle_vel = self.estimate_obstacle_velocity()

        with self.lock:
            pose = None if self.current_pose is None else self.current_pose.copy()
            obs = None if self.tracked_obstacle is None else self.tracked_obstacle.copy()

        robot_speed = min(math.hypot(robot_vel['vx'], robot_vel['vy']), self.supervisor_max_robot_speed)
        obstacle_speed = math.hypot(obstacle_vel['vx'], obstacle_vel['vy'])

        if pose is not None and obs is not None:
            dx = obs['x'] - pose['x']
            dy = obs['y'] - pose['y']
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                ux = dx / dist
                uy = dy / dist
                robot_toward = max(0.0, robot_vel['vx'] * ux + robot_vel['vy'] * uy)
                obstacle_toward = max(0.0, -(obstacle_vel['vx'] * ux + obstacle_vel['vy'] * uy))
                closing_speed = robot_toward + obstacle_toward
            else:
                closing_speed = robot_speed + obstacle_speed
        else:
            closing_speed = robot_speed + obstacle_speed

        motion_scale = 1.0
        if obstacle_speed < self.static_obstacle_speed_threshold:
            motion_scale = self.static_dynamic_margin_scale

        scan_period = 1.0 / max(self.scan_rate_hz, 1e-6)
        delay = scan_period + self.emergency_reaction_time * self.reaction_time_scale
        dynamic_margin = self.base_safety_margin + closing_speed * delay * motion_scale
        dynamic_margin = max(0.0, min(self.max_safety_margin, dynamic_margin))

        return self.obstacle_collision_radius + dynamic_margin

    def _compute_supervisor_metrics(self, pose: dict, obs: dict, robot_vel: dict, obstacle_vel: dict) -> dict:
        """Compute stopping-distance and TTC metrics for emergency supervision."""
        dx = obs['x'] - pose['x']
        dy = obs['y'] - pose['y']
        center_distance = math.hypot(dx, dy)

        combined_collision_radius = self.robot_collision_radius + self.obstacle_collision_radius
        clearance = max(0.0, center_distance - combined_collision_radius)

        robot_speed = math.hypot(robot_vel['vx'], robot_vel['vy'])
        if center_distance > 1e-6:
            ux = dx / center_distance
            uy = dy / center_distance
            rel_vx = robot_vel['vx'] - obstacle_vel['vx']
            rel_vy = robot_vel['vy'] - obstacle_vel['vy']
            closing_speed = max(0.0, rel_vx * ux + rel_vy * uy)
        else:
            closing_speed = robot_speed

        closing_speed = min(closing_speed, self.supervisor_max_closing_speed)

        brake_decel = max(self.supervisor_brake_deceleration, 1e-6)
        stopping_distance = (robot_speed * robot_speed) / (2.0 * brake_decel)
        latency_distance = robot_speed * self.supervisor_total_latency
        required_clearance = stopping_distance + latency_distance + self.supervisor_buffer

        # Static/near-static case: slightly relax required clearance to avoid
        # unnecessary early stop while preserving dynamic safety behavior.
        if closing_speed < self.supervisor_static_closing_threshold:
            required_clearance = max(0.0, required_clearance - self.supervisor_static_clearance_relax)

        if closing_speed <= 1e-4:
            ttc = float('inf')
        else:
            ttc = clearance / closing_speed

        trigger_now = (clearance <= required_clearance) or (ttc <= self.supervisor_ttc_critical)
        return {
            'center_distance': center_distance,
            'clearance': clearance,
            'closing_speed': closing_speed,
            'stopping_distance': stopping_distance,
            'latency_distance': latency_distance,
            'required_clearance': required_clearance,
            'ttc': ttc,
            'trigger_now': trigger_now,
        }

    def _apply_supervisor_hysteresis(self, metrics: dict) -> bool:
        """Apply hold/release hysteresis for stable emergency output."""
        now = self.get_clock().now().nanoseconds / 1e9
        trigger_now = metrics['trigger_now']

        if trigger_now:
            self.supervisor_emergency_active = True
            self.supervisor_last_trigger_time = now
            return True

        if not self.supervisor_emergency_active:
            return False

        # Keep emergency asserted for minimum hold time
        if (now - self.supervisor_last_trigger_time) < self.supervisor_min_hold_time:
            return True

        clear_release = metrics['clearance'] > (metrics['required_clearance'] + self.supervisor_release_margin)
        ttc_release = metrics['ttc'] > self.supervisor_ttc_release

        if clear_release and ttc_release:
            self.supervisor_emergency_active = False
            return False

        return True

    def _point_in_rectangle(self, px, py, corners):
        """Check if point (px, py) is inside rectangle defined by corners."""
        # Use cross product method
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign((px, py), corners[0], corners[1])
        d2 = sign((px, py), corners[1], corners[2])
        d3 = sign((px, py), corners[2], corners[3])
        d4 = sign((px, py), corners[3], corners[0])
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)
        
        return not (has_neg and has_pos)

    def _circle_intersects_segment(self, cx, cy, radius, p1, p2):
        """Check if circle intersects line segment."""
        # Vector from p1 to p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Vector from p1 to circle center
        fx = cx - p1[0]
        fy = cy - p1[1]
        
        # Project circle center onto line segment
        if dx == 0 and dy == 0:
            # Degenerate segment
            dist = math.hypot(fx, fy)
            return dist <= radius
        
        t = max(0, min(1, (fx * dx + fy * dy) / (dx * dx + dy * dy)))
        
        # Closest point on segment
        closest_x = p1[0] + t * dx
        closest_y = p1[1] + t * dy
        
        # Distance from circle center to closest point
        dist = math.hypot(cx - closest_x, cy - closest_y)
        
        return dist <= radius

    def collision_check_callback(self):
        """Periodic callback to check for future collisions along trajectory."""
        with self.lock:
            tracking = self.tracking
            obs = None if self.tracked_obstacle is None else self.tracked_obstacle.copy()
            pose = None if self.current_pose is None else self.current_pose.copy()

        if (not tracking) or obs is None or pose is None:
            self.publish_collision_visualization(None, None, [], [], False, None)
            flag_msg = Int32()
            flag_msg.data = 0
            self.close_pub.publish(flag_msg)
            self.supervisor_emergency_active = False
            return
        
        # tracked_obstacle is already in map frame - no conversion needed
        obs_map_x = obs['x']
        obs_map_y = obs['y']
        
        current_distance = math.hypot(obs_map_x - pose['x'], obs_map_y - pose['y'])
        
        # Only predict if obstacle is within detection threshold
        if current_distance > self.detection_threshold:
            self.publish_collision_visualization(None, None, [], [], False, None)
            # Clear emergency brake when obstacle is out of range
            flag_msg = Int32()
            flag_msg.data = 0
            self.close_pub.publish(flag_msg)
            return
        
        # Check collision at intermediate steps along the trajectory
        collision_detected_geom = False
        collision_time = None
        effective_collision_radius = (
            self.get_dynamic_collision_radius()
            if self.use_dynamic_radius_for_geometry
            else self.obstacle_collision_radius
        )
        
        robot_trajectory = []
        obstacle_trajectory = []

        # Check immediate collision at current timestep
        if self.check_rectangle_circle_collision(
            pose,
            {'x': obs_map_x, 'y': obs_map_y},
            collision_radius=effective_collision_radius,
        ):
            collision_detected_geom = True
            collision_time = 0.0
        
        # Check at prediction_steps intermediate points
        prediction_time = self.get_collision_prediction_time()
        if not collision_detected_geom:
            for step in range(1, self.prediction_steps + 1):
                t = (step / self.prediction_steps) * prediction_time

                robot_pred = self.predict_robot_position(t)
                obs_pred = self.predict_obstacle_position(t)

                if robot_pred is not None:
                    robot_trajectory.append(robot_pred)
                if obs_pred is not None:
                    obstacle_trajectory.append(obs_pred)

                # Check for collision at this time step
                if self.check_rectangle_circle_collision(
                    robot_pred,
                    obs_pred,
                    collision_radius=effective_collision_radius,
                ):
                    collision_detected_geom = True
                    collision_time = t
                    break  # Stop checking further if collision found

        robot_vel = self.estimate_robot_velocity()
        obstacle_vel = self.estimate_obstacle_velocity()
        supervisor_metrics = self._compute_supervisor_metrics(pose, obs, robot_vel, obstacle_vel)
        if self.use_supervisor_logic:
            collision_detected_supervisor = self._apply_supervisor_hysteresis(supervisor_metrics)
        else:
            collision_detected_supervisor = False

        # Geometry is a secondary guard. Gate it with supervisor context to avoid
        # early triggers when TTC is large/infinite and clearance is comfortably safe.
        geometry_guard_trigger = collision_detected_geom and (
            (supervisor_metrics['ttc'] <= self.supervisor_geometry_ttc_gate)
            or (
                supervisor_metrics['clearance']
                <= (supervisor_metrics['required_clearance'] + self.supervisor_geometry_clearance_gate)
            )
        )

        collision_detected = collision_detected_supervisor or geometry_guard_trigger

        if self.supervisor_debug_logs:
            now = self.get_clock().now().nanoseconds / 1e9
            if (now - self.supervisor_last_debug_log_time) >= self.supervisor_debug_log_period:
                self.supervisor_last_debug_log_time = now
                ttc_str = f"{supervisor_metrics['ttc']:.2f}" if math.isfinite(supervisor_metrics['ttc']) else 'inf'
                self.get_logger().info(
                    f"SUPERVISOR_METRICS clr={supervisor_metrics['clearance']:.3f}m "
                    f"req={supervisor_metrics['required_clearance']:.3f}m "
                    f"close_v={supervisor_metrics['closing_speed']:.3f}m/s "
                    f"ttc={ttc_str}s "
                    f"geom={int(collision_detected_geom)} geom_gate={int(geometry_guard_trigger)} "
                    f"sup={int(collision_detected_supervisor)} final={int(collision_detected)}"
                )
        
        # Publish emergency brake signal
        flag_msg = Int32()
        flag_msg.data = 1 if collision_detected else 0
        self.close_pub.publish(flag_msg)
        
        # Publish visualization
        self.publish_collision_visualization(
            pose, {'x': obs_map_x, 'y': obs_map_y},
            robot_trajectory, obstacle_trajectory,
            collision_detected, collision_time
        )
        # Publish custom EmergencyStop message
        emergency_msg = EmergencyStop()
        emergency_msg.emergency_triggered = 1 if collision_detected else 0
        if collision_detected:
            robot_speed = math.hypot(robot_vel['vx'], robot_vel['vy'])
            if collision_time is not None:
                travel_distance = robot_speed * collision_time
            else:
                travel_distance = supervisor_metrics['required_clearance']
            emergency_msg.distance = float(travel_distance)
            self.get_logger().warn(
                    f'🚨 EMERGENCY STOP TRIGGERED!\n'
                    f'  Travel distance: {emergency_msg.distance:.3f}m\n'
                    f'  Clearance: {supervisor_metrics["clearance"]:.3f}m '
                    f'(required: {supervisor_metrics["required_clearance"]:.3f}m), '
                    f'TTC: {supervisor_metrics["ttc"]:.2f}s\n'
                    f'  Obstacle: pos=({obs_map_x:.2f}, {obs_map_y:.2f}) '
                    f'My robot : x: {pose["x"]:.2f}, y: {pose["y"]:.2f}°\n'
                )
        else:
            emergency_msg.distance = 0.0
        self.emergency_stop_pub.publish(emergency_msg)

    def publish_collision_visualization(self, current_robot, current_obstacle,
                                       robot_trajectory, obstacle_trajectory,
                                       collision_detected, collision_time):
        """Publish visualization markers for collision geometry and trajectories."""
        marker_id = 0
        
        # Clear all markers if no data
        if current_robot is None:
            delete_marker = Marker()
            delete_marker.header.frame_id = 'map'
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.action = Marker.DELETEALL
            self.collision_viz_pub.publish(delete_marker)
            return

        effective_collision_radius = (
            self.get_dynamic_collision_radius()
            if self.use_dynamic_radius_for_geometry
            else self.obstacle_collision_radius
        )
        
        # Current robot rectangle (solid green)
        marker_id = self._publish_rectangle_marker(
            current_robot, marker_id, 'current_robot',
            (0.0, 1.0, 0.0, 0.5)  # Green, semi-transparent
        )
        
        # Current obstacle circle (solid blue)
        marker_id = self._publish_circle_marker(
            current_obstacle, marker_id, 'current_obstacle',
            (0.0, 0.0, 1.0, 0.5),  # Blue, semi-transparent
            radius=effective_collision_radius,
        )
        
        # Predicted trajectory - robot rectangles (faded)
        for i, robot_pos in enumerate(robot_trajectory):
            alpha = 0.3 - (i * 0.05)  # Fade out gradually
            if alpha < 0.1:
                alpha = 0.1
            color = (1.0, 0.0, 0.0, alpha) if collision_detected else (0.0, 1.0, 0.0, alpha)
            marker_id = self._publish_rectangle_marker(
                robot_pos, marker_id, f'predicted_robot_{i}', color
            )
        
        # Predicted trajectory - obstacle circles (faded)
        for i, obs_pos in enumerate(obstacle_trajectory):
            alpha = 0.3 - (i * 0.05)  # Fade out gradually
            if alpha < 0.1:
                alpha = 0.1
            color = (1.0, 0.0, 0.0, alpha) if collision_detected else (0.0, 0.0, 1.0, alpha)
            marker_id = self._publish_circle_marker(
                obs_pos, marker_id, f'predicted_obstacle_{i}', color,
                radius=effective_collision_radius,
            )

    def _publish_rectangle_marker(self, pose, marker_id, ns, color):
        """Publish a rectangle marker for the robot."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = pose['x']
        marker.pose.position.y = pose['y']
        marker.pose.position.z = 0.0
        
        # Set orientation from theta
        marker.pose.orientation.z = math.sin(pose['theta'] / 2.0)
        marker.pose.orientation.w = math.cos(pose['theta'] / 2.0)
        
        marker.scale.x = self.robot_length
        marker.scale.y = self.robot_width
        marker.scale.z = 0.1
        
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        
        try:
            from rclpy.duration import Duration
            marker.lifetime = Duration(seconds=0.2).to_msg()
        except Exception: 
            pass
        
        self.collision_viz_pub.publish(marker)
        return marker_id + 1

    def _publish_circle_marker(self, pos, marker_id, ns, color, radius: Optional[float] = None):
        """Publish a circle marker for the obstacle."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        marker.pose.position.x = pos['x']
        marker.pose.position.y = pos['y']
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        circle_radius = self.obstacle_collision_radius if radius is None else float(radius)
        marker.scale.x = circle_radius * 2.0
        marker.scale.y = circle_radius * 2.0
        marker.scale.z = 0.1
        
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        
        try:
            from rclpy.duration import Duration
            marker.lifetime = Duration(seconds=0.2).to_msg()
        except Exception:
            pass
        
        self.collision_viz_pub.publish(marker)
        return marker_id + 1

    #
    # Processing utilities (vectorized)
    #
    def scan_to_points(
        self, scan: LaserScan, downsample: int = 1, return_indices: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]: 
        ranges = np.array(scan.ranges, dtype=np.float32)
        n = ranges.size
        if n == 0:
            return (
                (np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32))
                if return_indices
                else np.empty((0, 2), dtype=np.float32)
            )

        angles = scan.angle_min + np.arange(n, dtype=np.float32) * scan.angle_increment
        valid_mask = (ranges > scan.range_min) & (ranges < scan.range_max) & np.isfinite(ranges)

        if self.min_point_distance is not None and self.min_point_distance > 0.0:
            valid_mask = valid_mask & (ranges >= float(self.min_point_distance))

        if downsample > 1:
            stride_mask = (np.arange(n) % downsample) == 0
            valid_mask = valid_mask & stride_mask

        if not np.any(valid_mask):
            return (
                (np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32))
                if return_indices
                else np.empty((0, 2), dtype=np.float32)
            )

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

        px = points[:, 0].astype(np.float64)
        py = points[:, 1].astype(np.float64)
        mx = pose['x'] + ct * px - st * py
        my = pose['y'] + st * px + ct * py

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
        # Keep points in free space (0) and unknown (-1), filter out occupied (>0, typically 100)
        free_mask[in_bounds] = (occ_vals == 0) | (occ_vals == -1)

        if not np.any(free_mask):
            return np.empty((0, 2), dtype=np.float32)

        filtered_points = points[free_mask]
        if indices is not None and indices.size == points.shape[0]:
            filtered_indices = indices[free_mask]
            order = np.argsort(filtered_indices)
            filtered_points = filtered_points[order]

        return filtered_points

    def is_valid_map_position(self, map_x: float, map_y: float) -> bool:
        """
        Check if a position in MAP frame is in a valid area.
        Returns True if position is inside map bounds AND in free/unknown space.
        Returns False if outside map bounds OR in occupied space.
        """
        if self.map_np is None or not self.map_info:
            return True  # No map available, assume valid
        
        res = self.map_info['res']
        width = self.map_info['width']
        height = self.map_info['height']
        ox = self.map_info['ox']
        oy = self.map_info['oy']
        yaw = self.map_info.get('yaw', 0.0)
        
        # Transform from map frame to occupancy grid indices
        dx = map_x - ox
        dy = map_y - oy
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        grid_x = cos_y * dx + sin_y * dy
        grid_y = -sin_y * dx + cos_y * dy
        
        cx = int(math.floor(grid_x / res))
        cy = int(math.floor(grid_y / res))
        
        # Check bounds - positions outside map are INVALID
        if cx < 0 or cx >= width or cy < 0 or cy >= height:
            return False
        
        # Check occupancy - occupied (>0, typically 100) is INVALID
        occ_val = self.map_np[cy, cx]
        return occ_val == 0 or occ_val == -1

    def cluster_scan_ordered(
        self, points: np.ndarray, min_points: int = 3, max_gap: float = 0.1, max_diameter: Optional[float] = None
    ):
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
                            diffs = cluster_arr[: , None, :] - cluster_arr[None, :, :]
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
                    diffs = cluster_arr[: , None, :] - cluster_arr[None, :, :]
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
        """Publish obstacle position. obs is in MAP frame."""
        self._pose_msg.header.frame_id = 'map'
        self._pose_msg.pose.position.x = float(obs['x'])
        self._pose_msg.pose.position.y = float(obs['y'])
            
        self._pose_msg.header.stamp = self.get_clock().now().to_msg()
        self._pose_msg.pose.position.z = 0.0
        self._pose_msg.pose.orientation.x = 0.0
        self._pose_msg.pose.orientation.y = 0.0
        self._pose_msg.pose.orientation.z = 0.0
        self._pose_msg.pose.orientation.w = 1.0
        self.obstacle_pub.publish(self._pose_msg)

        obstacle_vel = self.estimate_obstacle_velocity()
        self._twist_msg.linear.x = float(obstacle_vel['vx'])
        self._twist_msg.linear.y = float(obstacle_vel['vy'])
        self._twist_msg.linear.z = 0.0
        self._twist_msg.angular.x = 0.0
        self._twist_msg.angular.y = 0.0
        self._twist_msg.angular.z = 0.0
        self.obstacle_speed_pub.publish(self._twist_msg)

    def publish_marker(self, obs: dict):
        """Publish obstacle marker. obs is in MAP frame."""
        m = self._marker_msg
        m.header.frame_id = 'map'
        m.pose.position.x = float(obs['x'])
        m.pose.position.y = float(obs['y'])
        m.pose.position.z = 0.0

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

    def _publish_delete_marker(self):
        m = self._delete_marker
        if self.current_pose is not None:
            m.header.frame_id = 'map'
        else:
            m.header.frame_id = 'base_link'
        m.header.stamp = self.get_clock().now().to_msg()
        self.marker_pub.publish(m)

    def delete_marker(self):
        self._publish_delete_marker()

    def _send_transform(self, pose: dict, stamp=None):
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
        self.tf_broadcaster.sendTransform(t)

    def destroy_node(self):
        self._publish_delete_marker()
        try:
            self.pool.shutdown(wait=False)
        except Exception:
            pass
        super().destroy_node()


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