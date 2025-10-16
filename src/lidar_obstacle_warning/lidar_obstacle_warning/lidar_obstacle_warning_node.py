#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
import math
import numpy as np

class ObstacleTracker(Node):
    def __init__(self):
        super().__init__('obstacle_tracker')
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        self.create_subscription(Float32MultiArray, '/robot_pose_raw', self.pose_callback, 10)
        self.obstacle_pub = self.create_publisher(PoseStamped, '/obstacle_relative', 10)
        self.close_pub = self.create_publisher(PoseStamped, '/obstacle_too_close', 10)
        self.marker_pub = self.create_publisher(Marker, '/obstacle_candidate_marker', 10)
        
        self.latest_scan = None
        self.latest_map = None
        self.current_pose = None
        self.tracked_obstacle = None
        self.tracking = False
        self.close_threshold = 0.5
        self.cluster_eps = 0.1
        self.cluster_min_points = 3

    def scan_callback(self, msg):
        self.latest_scan = msg
        if self.tracking and self.tracked_obstacle is not None:
            points = self.scan_to_points(msg)
            if len(points) > 0:
                clusters = self.cluster_points(points)
                if clusters:
                    tracked_pos = np.array([self.tracked_obstacle['x'], self.tracked_obstacle['y']])
                    dists = [np.linalg.norm(np.mean(c, axis=0) - tracked_pos) for c in clusters]
                    closest_idx = np.argmin(dists)
                    closest_cluster = clusters[closest_idx]
                    centroid = np.mean(closest_cluster, axis=0)
                    self.tracked_obstacle = {'x': centroid[0], 'y': centroid[1]}
                    self.publish_obstacle(self.tracked_obstacle)
                    self.publish_marker(self.tracked_obstacle)
                else:
                    self.tracked_obstacle = None
                    self.tracking = False
                    self.delete_marker()
        else:
            self.delete_marker()

    def map_callback(self, msg):
        self.latest_map = msg

    def pose_callback(self, msg):
        if len(msg.data) < 3:
            self.get_logger().warn('Invalid pose data received')
            return
        self.current_pose = {'x': msg.data[0], 'y': msg.data[1], 'theta': msg.data[2]}
        
        # Reset tracking and delete marker
        self.tracking = False
        self.tracked_obstacle = None
        self.delete_marker()
        
        if self.latest_scan and self.latest_map:
            points = self.scan_to_points(self.latest_scan)
            if len(points) > 0:
                filtered_points = self.filter_free_space(points, self.current_pose, self.latest_map)
                if len(filtered_points) > 0:
                    clusters = self.cluster_points(filtered_points)
                    if clusters:
                        largest_cluster = max(clusters, key=len)
                        centroid = np.mean(largest_cluster, axis=0)
                        self.tracked_obstacle = {'x': centroid[0], 'y': centroid[1]}
                        self.tracking = True
                        self.publish_obstacle(self.tracked_obstacle)
                        self.publish_marker(self.tracked_obstacle)
                    else:
                        self.delete_marker()

    def scan_to_points(self, scan):
        points = []
        angle = scan.angle_min
        for r in scan.ranges:
            if scan.range_min < r < scan.range_max:
                px = -r * math.cos(angle)
                py = -r * math.sin(angle)
                points.append([px, py])
            angle += scan.angle_increment
        return np.array(points)

    def filter_free_space(self, points, pose, map_msg):
        filtered = []
        res = map_msg.info.resolution
        width = map_msg.info.width
        height = map_msg.info.height
        ox = map_msg.info.origin.position.x
        oy = map_msg.info.origin.position.y
        ct = math.cos(pose['theta'])
        st = math.sin(pose['theta'])
        for p in points:
            mx = pose['x'] + p[0] * ct - p[1] * st
            my = pose['y'] + p[1] * ct + p[0] * st
            cx = int((mx - ox) / res)
            cy = int((my - oy) / res)
            if 0 <= cx < width and 0 <= cy < height:
                idx = cy * width + cx
                if map_msg.data[idx] == 0:
                    filtered.append(p)
        return np.array(filtered)

    def cluster_points(self, points):
        if len(points) == 0:
            return []
        clusters = []
        for p in points:
            p_arr = np.array(p)
            found = False
            for i, c in enumerate(clusters):
                dists = np.linalg.norm(np.array(c) - p_arr, axis=1)
                if np.min(dists) < self.cluster_eps:
                    clusters[i].append(p)
                    found = True
                    break
            if not found:
                clusters.append([p])
        clusters = [np.array(c) for c in clusters if len(c) >= self.cluster_min_points]
        return clusters

    def publish_obstacle(self, obs):
        pose_st = PoseStamped()
        pose_st.header.stamp = self.get_clock().now().to_msg()
        pose_st.header.frame_id = 'base_link'
        pose_st.pose.position.x = obs['x']
        pose_st.pose.position.y = obs['y']
        pose_st.pose.position.z = 0.0
        pose_st.pose.orientation.w = 1.0
        self.obstacle_pub.publish(pose_st)
        
        dist = math.sqrt(obs['x']**2 + obs['y']**2)
        if dist < self.close_threshold:
            self.close_pub.publish(pose_st)

    def publish_marker(self, obs):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'base_link'
        marker.ns = 'obstacle'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = obs['x']
        marker.pose.position.y = obs['y']
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        # Set marker lifetime to 1 second
        marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()
        self.marker_pub.publish(marker)

    def delete_marker(self):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'base_link'
        marker.ns = 'obstacle'
        marker.id = 0
        marker.action = Marker.DELETE
        self.marker_pub.publish(marker)

    def destroy_node(self):
        self.delete_marker()
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