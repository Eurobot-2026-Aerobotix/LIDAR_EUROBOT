#!/usr/bin/env python3
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List
from collections import deque

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


class ObstacleTracker(Node):
    """
    Obstacle detection node for clustering LIDAR scans and publishing obstacles.
    Includes collision prediction with trajectory forecasting.
    """

    def __init__(self):
        super().__init__('obstacle_tracker')

        # ====== COLLISION PREDICTION PARAMETERS ======
        # Robot collision geometry (rectangle)
        self.robot_length = 0.3  # meters
        self.robot_width = 0.3   # meters
        
        # Obstacle collision geometry (circle)
        self.obstacle_collision_radius = 0.25  # meters
        
        # Detection threshold - start prediction when obstacle is within this distance
        self.detection_threshold = 1.0  # meters
        
        # Collision prediction time horizon (accounts for system latency)
        # Observed latency:  0.1m at 0.7m/s = ~0.143s
        self.collision_prediction_time = 0.143  # seconds
        
        # Number of intermediate steps to check along trajectory
        self.prediction_steps = 3  # Check 3 intermediate positions + final position
        
        # History length for velocity estimation
        self.history_length = 5
        
        # Velocity histories
        self.robot_pose_history = deque(maxlen=self.history_length)
        self.obstacle_position_history = deque(maxlen=self.history_length)
        
        # Timer for periodic collision checking
        self.collision_check_timer = self.create_timer(0.1, self.collision_check_callback)  # 10 Hz
        # ============================================

        # Original parameters
        self.scan_skip = 2
        self.range_downsample = 1
        self.cluster_min_points = 5
        self.cluster_min_distance = 0.1
        self.cluster_max_diameter = 0.3
        self.close_threshold = 0.5
        self.scan_tracking_stop_distance = 0.8
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
        self.close_pub = self.create_publisher(Int32, '/emergency_break', 10)
        self.marker_pub = self.create_publisher(Marker, '/obstacle_candidate_marker', 10)
        
        # New publishers for collision visualization
        self.collision_viz_pub = self.create_publisher(Marker, '/collision_visualization', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Cached state
        self.latest_scan:  Optional[LaserScan] = None
        self.latest_map_msg: Optional[OccupancyGrid] = None
        self.map_np:  Optional[np.ndarray] = None
        self.map_info = {}
        self.current_pose = None
        self.tracked_obstacle = None
        self.tracking = False

        # Helpers
        self.scan_counter = 0
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=2)

        # Reused message objects
        self._pose_msg = PoseStamped()
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
            f'  Prediction time:  {self.collision_prediction_time}s\n'
            f'  Prediction steps: {self.prediction_steps}'
        )

    #
    # Callbacks
    #
    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        with self.lock:
            self.scan_counter += 1
            current_scan_num = int(self.scan_counter)

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
        with self.lock:
            tracking = self.tracking
            tracked_exists = self.tracked_obstacle is not None
            current_pose_copy = None if self.current_pose is None else self.current_pose.copy()

        if not tracking or not tracked_exists:
            return

        pts, _ = self.scan_to_points(scan, downsample=self.range_downsample, return_indices=True)
        if pts.size == 0:
            return

        accepted_clusters, _ = self.cluster_scan_ordered(
            pts,
            min_points=self.cluster_min_points,
            max_gap=self.cluster_min_distance,
            max_diameter=self.cluster_max_diameter,
        )

        if not accepted_clusters:
            return

        with self.lock:
            prev = np.array([self.tracked_obstacle['x'], self.tracked_obstacle['y']])
        centroids = [np.mean(c, axis=0) for c in accepted_clusters]
        dists = [np.linalg.norm(c - prev) for c in centroids]
        best_idx = int(np.argmin(dists))
        best_centroid = centroids[best_idx]
        best_dist = dists[best_idx]

        if best_dist > self.scan_tracking_stop_distance: 
            with self.lock:
                self.tracking = False
                self.tracked_obstacle = None
            self._publish_delete_marker()
            return

        with self.lock:
            self.tracked_obstacle = {'x': float(best_centroid[0]), 'y': float(best_centroid[1])}
            timestamp = self.get_clock().now().nanoseconds / 1e9
            self.obstacle_position_history.append({
                'x': self.tracked_obstacle['x'],
                'y': self.tracked_obstacle['y'],
                'timestamp': timestamp
            })

        self.publish_obstacle(self.tracked_obstacle)
        self.publish_marker(self.tracked_obstacle, use_map=False)

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
        pose = {'x': float(msg.x), 'y': float(msg.y), 'theta': float(msg.z)}
        
        with self.lock:
            self.tracking = False
            self.tracked_obstacle = None
            self.current_pose = pose
            timestamp = self.get_clock().now().nanoseconds / 1e9
            self.robot_pose_history.append({
                'x': pose['x'],
                'y': pose['y'],
                'theta': pose['theta'],
                'timestamp': timestamp
            })

        self._send_transform(pose)

        if self.latest_scan is None or self.latest_map_msg is None:
            return

        with self.lock:
            scan_num = int(self.scan_counter)

        try:
            self.pool.submit(self._process_pose_trigger, self.latest_scan, pose, scan_num)
        except Exception:
            self._process_pose_trigger(self.latest_scan, pose, scan_num)

    def _process_pose_trigger(self, scan: LaserScan, pose: dict, scan_num: int):
        pts, idxs = self.scan_to_points(scan, downsample=self.range_downsample, return_indices=True)
        if pts.size == 0:
            self._publish_delete_marker()
            return

        clusters, _ = self.cluster_scan_ordered(
            pts,
            min_points=self.cluster_min_points,
            max_gap=self.cluster_min_distance,
            max_diameter=None,
        )

        accepted_clusters:  List[np.ndarray] = []
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
                    diffs = filtered[:, None, :] - filtered[None, : , :]
                    dists = np.hypot(diffs[..., 0], diffs[..., 1])
                    diam = float(dists.max())
                if diam > self.cluster_max_diameter:
                    continue
            accepted_clusters.append(filtered)

        if not accepted_clusters: 
            self._publish_delete_marker()
            return

        largest = max(accepted_clusters, key=lambda c: c.shape[0])
        centroid = np.mean(largest, axis=0)

        with self.lock:
            self.tracked_obstacle = {'x':  float(centroid[0]), 'y': float(centroid[1])}
            self.tracking = True
            self.current_pose = pose
            timestamp = self.get_clock().now().nanoseconds / 1e9
            self.obstacle_position_history.append({
                'x': self.tracked_obstacle['x'],
                'y': self.tracked_obstacle['y'],
                'timestamp': timestamp
            })

        self.publish_obstacle(self.tracked_obstacle)
        self.publish_marker(self.tracked_obstacle)

    #
    # Collision Prediction Methods
    #
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
        """Estimate obstacle velocity from position history (convert to map frame)."""
        with self.lock:
            if len(self.obstacle_position_history) < 2:
                return {'vx': 0.0, 'vy': 0.0}
            history = list(self.obstacle_position_history)
            current_pose = self.current_pose.copy() if self.current_pose else None
        
        if current_pose is None:
            return {'vx':  0.0, 'vy': 0.0}
        
        # Convert obstacle positions from robot frame to map frame
        map_positions = []
        for h in history:
            ct = math.cos(current_pose['theta'])
            st = math.sin(current_pose['theta'])
            mx = current_pose['x'] + ct * h['x'] - st * h['y']
            my = current_pose['y'] + st * h['x'] + ct * h['y']
            map_positions.append({'x': mx, 'y': my, 'timestamp': h['timestamp']})
        
        dt = map_positions[-1]['timestamp'] - map_positions[0]['timestamp']
        if dt < 1e-6:
            return {'vx': 0.0, 'vy': 0.0}
        
        dx = map_positions[-1]['x'] - map_positions[0]['x']
        dy = map_positions[-1]['y'] - map_positions[0]['y']
        
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
            if self.tracked_obstacle is None or self.current_pose is None:
                return None
            obs = self.tracked_obstacle.copy()
            pose = self.current_pose.copy()
        
        vel = self.estimate_obstacle_velocity()
        
        # Convert obstacle from robot frame to map frame
        ct = math.cos(pose['theta'])
        st = math.sin(pose['theta'])
        obs_map_x = pose['x'] + ct * obs['x'] - st * obs['y']
        obs_map_y = pose['y'] + st * obs['x'] + ct * obs['y']
        
        # Predict future position
        predicted_x = obs_map_x + vel['vx'] * time_ahead
        predicted_y = obs_map_y + vel['vy'] * time_ahead
        
        return {
            'x': predicted_x,
            'y': predicted_y
        }

    def check_rectangle_circle_collision(self, robot_pos, obstacle_pos):
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
        radius = self.obstacle_collision_radius
        
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
            if not self.tracking or self.tracked_obstacle is None or self.current_pose is None:
                # Clear visualization if not tracking
                self.publish_collision_visualization(None, None, [], [], False, None)
                return
            obs = self.tracked_obstacle.copy()
            pose = self.current_pose.copy()
        
        # Convert obstacle to map frame for distance check
        ct = math.cos(pose['theta'])
        st = math.sin(pose['theta'])
        obs_map_x = pose['x'] + ct * obs['x'] - st * obs['y']
        obs_map_y = pose['y'] + st * obs['x'] + ct * obs['y']
        
        current_distance = math.hypot(obs_map_x - pose['x'], obs_map_y - pose['y'])
        
        # Only predict if obstacle is within detection threshold
        if current_distance > self.detection_threshold:
            self.publish_collision_visualization(None, None, [], [], False, None)
            return
        
        # Check collision at intermediate steps along the trajectory
        collision_detected = False
        collision_time = None
        
        robot_trajectory = []
        obstacle_trajectory = []
        
        # Check at prediction_steps intermediate points
        for step in range(1, self.prediction_steps + 1):
            t = (step / self.prediction_steps) * self.collision_prediction_time
            
            robot_pred = self.predict_robot_position(t)
            obs_pred = self.predict_obstacle_position(t)
            
            if robot_pred is not None: 
                robot_trajectory.append(robot_pred)
            if obs_pred is not None:
                obstacle_trajectory.append(obs_pred)
            
            # Check for collision at this time step
            if self.check_rectangle_circle_collision(robot_pred, obs_pred):
                collision_detected = True
                collision_time = t
                break  # Stop checking further if collision found
        
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
        
        if collision_detected:
            self.get_logger().warn(
                f'Collision predicted in {collision_time:.3f}s!  Emergency brake activated.'
            )

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
        
        # Current robot rectangle (solid green)
        marker_id = self._publish_rectangle_marker(
            current_robot, marker_id, 'current_robot',
            (0.0, 1.0, 0.0, 0.5)  # Green, semi-transparent
        )
        
        # Current obstacle circle (solid blue)
        marker_id = self._publish_circle_marker(
            current_obstacle, marker_id, 'current_obstacle',
            (0.0, 0.0, 1.0, 0.5)  # Blue, semi-transparent
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
                obs_pos, marker_id, f'predicted_obstacle_{i}', color
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

    def _publish_circle_marker(self, pos, marker_id, ns, color):
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
        
        marker.scale.x = self.obstacle_collision_radius * 2.0
        marker.scale.y = self.obstacle_collision_radius * 2.0
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
        free_mask[in_bounds] = np.isin(occ_vals, [0, -1])

        if not np.any(free_mask):
            return np.empty((0, 2), dtype=np.float32)

        filtered_points = points[free_mask]
        if indices is not None and indices.size == points.shape[0]:
            filtered_indices = indices[free_mask]
            order = np.argsort(filtered_indices)
            filtered_points = filtered_points[order]

        return filtered_points

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
        if self.current_pose is not None:
            pose = self.current_pose
            ct = math.cos(pose['theta'])
            st = math.sin(pose['theta'])
            mx = pose['x'] + ct * obs['x'] - st * obs['y']
            my = pose['y'] + st * obs['x'] + ct * obs['y']
            self._pose_msg.header.frame_id = 'map'
            self._pose_msg.pose.position.x = float(mx)
            self._pose_msg.pose.position.y = float(my)
        else:
            self.get_logger().warning('No robot pose available')
            
        self._pose_msg.header.stamp = self.get_clock().now().to_msg()
        self._pose_msg.pose.position.z = 0.0
        self._pose_msg.pose.orientation.x = 0.0
        self._pose_msg.pose.orientation.y = 0.0
        self._pose_msg.pose.orientation.z = 0.0
        self._pose_msg.pose.orientation.w = 1.0
        self.obstacle_pub.publish(self._pose_msg)

    def publish_marker(self, obs: dict, use_map:  bool = True):
        m = self._marker_msg
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
            m.header.frame_id = 'base_link'
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