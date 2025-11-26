#!/usr/bin/env python3
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3, TransformStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster
from rclpy.node import Node
from std_msgs.msg import Int32


class ObstacleTracker(Node):
    """
    Obstacle detection node that:
    - Converts scans to points (vectorized),
    - Filters points using a cached numpy map,
    - Clusters points using a scan-order O(n) algorithm,
    - Publishes a marker in the 'map' frame when a robot pose is available,
    - Publishes a TF (map -> base_link) when a new robot pose is received (event-driven)
      and also stamped with scan timestamps when scans arrive so TF consumers can
      transform scans that fall between pose updates.
    """

    def __init__(self):
        super().__init__('obstacle_tracker')

        # Parameters (tweak these to tune perf/behavior)
        self.scan_skip = 1  # process every Nth scan
        self.range_downsample = 1  # downsample LIDAR ranges (1 = keep all, 2 = every other)
        self.cluster_min_points = 5
        self.cluster_min_distance = 0.04  # max distance between consecutive points in a cluster
        self.close_threshold = 0.5

        # Subscriptions / publishers
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
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
    # Callbacks
    #
    def scan_callback(self, msg: LaserScan):
        # Keep the latest scan
        self.latest_scan = msg
        self.scan_counter += 1

        # If we have a known robot pose, re-publish the map->base_link transform
        # stamped with the scan's header.stamp so TF consumers can transform this scan.
        with self.lock:
            pose_for_tf = None if self.current_pose is None else self.current_pose.copy()

        if pose_for_tf is not None:
            # publish transform stamped at scan time (use scan.header.stamp if present)
            stamp = None
            try:
                # header.stamp is a builtin_interfaces.msg.Time in ROS2
                stamp = msg.header.stamp
            except Exception:
                stamp = None 
            self._send_transform(pose_for_tf, stamp=stamp)

        # Throttle processing: only handle every self.scan_skip scans
        if (self.scan_counter - 1) % self.scan_skip != 0:
            return

        try:
            self.pool.submit(self._process_scan, msg)
        except Exception:
            self._process_scan(msg)

    def _process_scan(self, scan: LaserScan):
        # Update location of currently tracked obstacle (if any)
        with self.lock:
            tracking = self.tracking
            tracked_exists = self.tracked_obstacle is not None

        if not tracking or not tracked_exists:
            return

        # Use only the scan for scan-time updates (do NOT consult the map/pose here).
        # Keep tracking state as-is; do not reset on a scan miss — resets happen only
        # in the pose callback per the requested behavior.
        pts, _ = self.scan_to_points(scan, downsample=self.range_downsample, return_indices=True)
        if pts.size == 0:
            # No points in this scan — keep existing tracking unchanged.
            return

        # Cluster directly on raw scan points (scan-ordered). These points are
        # expressed in the robot frame (base_link), which matches how
        # tracked_obstacle is stored when set from the pose-triggered detection.
        clusters = self.cluster_scan_ordered(pts,
                                            min_points=self.cluster_min_points,
                                            max_gap=self.cluster_min_distance)
        if not clusters:
            # No clusters found in this scan. Do not clear the tracked obstacle here;
            # keep tracking active until the next pose-triggered re-evaluation.
            return

        # Find the cluster whose centroid is closest to the previously tracked obstacle
        # (all in robot frame coordinates).
        with self.lock:
            prev = np.array([self.tracked_obstacle['x'], self.tracked_obstacle['y']])
        centroids = [np.mean(c, axis=0) for c in clusters]
        dists = [np.linalg.norm(c - prev) for c in centroids]
        best_idx = int(np.argmin(dists))
        best_centroid = centroids[best_idx]

        with self.lock:
            self.tracked_obstacle = {'x': float(best_centroid[0]), 'y': float(best_centroid[1])}

        # Publish the obstacle in the robot frame for scan-based updates (avoid using map/pose)
        self.publish_obstacle(self.tracked_obstacle)
        # When updating from scans we should publish the marker in base_link (robot frame)
        # so the marker position doesn't rely on the map/pose.
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
        self.map_info = {
            'res': info.resolution,
            'width': width,
            'height': height,
            'ox': info.origin.position.x,
            'oy': info.origin.position.y,
        }

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

        # Do NOT delete marker immediately here. We'll delete only if detection fails
        # (deleting here caused the marker to be removed before the detection could publish).
        # self._publish_delete_marker()

        if self.latest_scan is None or self.latest_map_msg is None:
            return

        try:
            self.pool.submit(self._process_pose_trigger, self.latest_scan, pose)
        except Exception:
            self._process_pose_trigger(self.latest_scan, pose)

    def _process_pose_trigger(self, scan: LaserScan, pose: dict):
        pts, idxs = self.scan_to_points(scan, downsample=self.range_downsample, return_indices=True)
        if pts.size == 0:
            # no points -> ensure marker removed
            self._publish_delete_marker()
            return

        filtered_pts = self.filter_free_space(pts, pose, idxs)
        if filtered_pts.size == 0:
            # no valid points -> ensure marker removed
            self._publish_delete_marker()
            return

        clusters = self.cluster_scan_ordered(filtered_pts,
                                            min_points=self.cluster_min_points,
                                            max_gap=self.cluster_min_distance)
        if not clusters:
            # no clusters -> ensure marker removed
            self._publish_delete_marker()
            return

        largest = max(clusters, key=lambda c: c.shape[0])
        centroid = np.mean(largest, axis=0)

        with self.lock:
            self.tracked_obstacle = {'x': float(centroid[0]), 'y': float(centroid[1])}
            self.tracking = True
            self.current_pose = pose

        self.publish_obstacle(self.tracked_obstacle)
        self.publish_marker(self.tracked_obstacle)

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

        ct = math.cos(pose['theta'])
        st = math.sin(pose['theta'])

        px = points[:, 0].astype(np.float64)
        py = points[:, 1].astype(np.float64)

        mx = pose['x'] + ct * px - st * py
        my = pose['y'] + st * px + ct * py

        cx = np.floor((mx - ox) / res).astype(np.int32)
        cy = np.floor((my - oy) / res).astype(np.int32)

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

    def cluster_scan_ordered(self, points: np.ndarray, min_points: int = 3, max_gap: float = 0.1):
        if points.size == 0:
            return []

        clusters = []
        current = [points[0]]
        for i in range(1, points.shape[0]):
            p_prev = points[i - 1]
            p_cur = points[i]
            if np.linalg.norm(p_cur - p_prev) <= max_gap:
                current.append(p_cur)
            else:
                if len(current) >= min_points:
                    clusters.append(np.array(current))
                current = [p_cur]
        if len(current) >= min_points:
            clusters.append(np.array(current))
        return clusters

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
        if dist < self.close_threshold:
            self.close_pub.publish(self._pose_msg)
            flag_msg.data = 1
        else:
            flag_msg.data = 0
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
