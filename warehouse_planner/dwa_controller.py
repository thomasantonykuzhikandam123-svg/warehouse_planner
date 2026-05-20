#!/usr/bin/env python3
"""
Regulated Pure Pursuit with rotate-to-heading prelude.

Reference:
- Coulter (1992) Pure Pursuit
- Macenski et al. (2023) Nav2 Regulated Pure Pursuit
- Nav2 design pattern: rotate-to-heading before path following
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path, OccupancyGrid
import math


class PathFollower(Node):

    def __init__(self):
        super().__init__('dwa_controller')

        # Robot limits
        self.max_v = 0.4
        self.min_v = 0.08
        self.max_omega = 1.5
        self.robot_radius = 0.22

        # Pure Pursuit lookahead
        self.lookahead_min = 0.5
        self.lookahead_max = 1.2
        self.lookahead_gain = 1.0

        # Regulation
        self.curvature_threshold = 0.6
        self.proximity_threshold = 0.5

        # Goal
        self.goal_tolerance = 0.4

        # Rotate-to-heading prelude (Nav2 pattern)
        self.heading_align_threshold = math.radians(35)  # turn first if >35 deg off
        self.heading_align_omega = 0.8

        # State
        self.current_path = []
        self.path_index = 0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.current_v = 0.0
        self.have_odom = False
        self.arrived = True
        self.aligning_heading = False   # NEW: are we in rotate-to-heading mode?

        # Map
        self.map_data = None
        self.map_resolution = 0.2
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_width = 0
        self.map_height = 0

        self.path_sub = self.create_subscription(Path, '/planned_path', self.path_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info('Pure Pursuit + rotate-to-heading prelude ready')

    # =====================================================
    # CALLBACKS
    # =====================================================
    def path_callback(self, msg):
        self.current_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.path_index = 0
        if self.current_path:
            self.arrived = False
            # ENTER rotate-to-heading mode on every new path
            self.aligning_heading = True
            self.get_logger().info(
                f'New path: {len(self.current_path)} points. Aligning heading first.')

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_theta = math.atan2(siny, cosy)
        self.current_v = msg.twist.twist.linear.x
        self.have_odom = True

    def map_callback(self, msg):
        self.map_data = list(msg.data)
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.map_width = msg.info.width
        self.map_height = msg.info.height

    # =====================================================
    # MAP UTILITIES
    # =====================================================
    def is_occupied(self, x, y):
        if self.map_data is None:
            return False
        gx = int((x - self.map_origin_x) / self.map_resolution)
        gy = int((y - self.map_origin_y) / self.map_resolution)
        if gx < 0 or gy < 0 or gx >= self.map_width or gy >= self.map_height:
            return True
        return self.map_data[gy * self.map_width + gx] > 50

    def footprint_collides(self, x, y):
        r = self.robot_radius
        offsets = [(0, 0), (r, 0), (-r, 0), (0, r), (0, -r)]
        for dx, dy in offsets:
            if self.is_occupied(x + dx, y + dy):
                return True
        return False

    def nearest_obstacle(self, x, y, max_d=0.8):
        n = int(max_d / self.map_resolution)
        for ring in range(1, n + 1):
            d = ring * self.map_resolution
            for ang_deg in range(0, 360, 45):
                ang = math.radians(ang_deg)
                if self.is_occupied(x + d * math.cos(ang), y + d * math.sin(ang)):
                    return d
        return max_d

    def will_collide_along_arc(self, v, omega, time_horizon):
        if v < 0.01:
            return False
        x, y, theta = self.robot_x, self.robot_y, self.robot_theta
        dt = 0.1
        steps = max(1, int(time_horizon / dt))
        for _ in range(steps):
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
            theta += omega * dt
            if self.footprint_collides(x, y):
                return True
        return False

    # =====================================================
    # MAIN CONTROL
    # =====================================================
    def control_loop(self):
        if not self.have_odom or not self.current_path or self.arrived:
            return
        if self.map_data is None:
            return

        # Goal arrival check
        goal_x, goal_y = self.current_path[-1]
        if math.hypot(goal_x - self.robot_x, goal_y - self.robot_y) < self.goal_tolerance:
            self.get_logger().info('Goal reached')
            self.stop_robot()
            self.arrived = True
            self.aligning_heading = False
            return

        # ===== ROTATE-TO-HEADING PRELUDE =====
        # Only at start of new path. Once aligned, switch to Pure Pursuit.
        if self.aligning_heading:
            # Find a path point about 1m ahead to determine initial heading
            target = self.find_initial_target_point()
            if target is None:
                self.aligning_heading = False
                return

            dx = target[0] - self.robot_x
            dy = target[1] - self.robot_y
            target_heading = math.atan2(dy, dx)
            heading_error = self.normalize_angle(target_heading - self.robot_theta)

            if abs(heading_error) < self.heading_align_threshold:
                # Aligned enough — switch to Pure Pursuit
                self.aligning_heading = False
                self.get_logger().info('Heading aligned, starting Pure Pursuit')
            else:
                # Rotate in place
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = math.copysign(self.heading_align_omega, heading_error)
                self.cmd_pub.publish(cmd)
                return

        # ===== PURE PURSUIT FOLLOWING =====

        # Monotonic closest-point-ahead
        best_i = self.path_index
        best_d = float('inf')
        end_idx = min(self.path_index + 25, len(self.current_path))
        for i in range(self.path_index, end_idx):
            d = math.hypot(self.current_path[i][0] - self.robot_x,
                           self.current_path[i][1] - self.robot_y)
            if d < best_d:
                best_d = d
                best_i = i
        self.path_index = best_i

        # Lookahead
        L = max(self.lookahead_min,
                min(self.lookahead_max,
                    self.lookahead_gain * max(self.current_v, 0.2)))

        lookahead = None
        for i in range(self.path_index, len(self.current_path)):
            d = math.hypot(self.current_path[i][0] - self.robot_x,
                           self.current_path[i][1] - self.robot_y)
            if d >= L:
                lookahead = self.current_path[i]
                break
        if lookahead is None:
            lookahead = self.current_path[-1]

        # Pure Pursuit math
        dx = lookahead[0] - self.robot_x
        dy = lookahead[1] - self.robot_y
        alpha = self.normalize_angle(math.atan2(dy, dx) - self.robot_theta)
        L_actual = math.hypot(dx, dy)
        curvature = 2.0 * math.sin(alpha) / L_actual if L_actual > 0.01 else 0.0

        # Velocity regulation
        v = self.max_v
        if abs(curvature) > self.curvature_threshold:
            v *= self.curvature_threshold / abs(curvature)
        near = self.nearest_obstacle(self.robot_x, self.robot_y)
        if near < self.proximity_threshold:
            v *= near / self.proximity_threshold
        v = max(self.min_v, min(self.max_v, v))

        omega = curvature * v
        omega = max(-self.max_omega, min(self.max_omega, omega))

        # Time-scaled collision check
        collision_horizon = min(0.4, L / max(v, 0.2))
        if self.will_collide_along_arc(v, omega, collision_horizon):
            v = self.min_v
            omega = curvature * v
            omega = max(-self.max_omega, min(self.max_omega, omega))
            if self.will_collide_along_arc(v, omega, collision_horizon):
                v = 0.0
                omega = 0.0
                self.get_logger().warn('Path blocked — stopping')

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega
        self.cmd_pub.publish(cmd)

    def find_initial_target_point(self):
        """Find a path point ~1m ahead for initial heading alignment."""
        if not self.current_path:
            return None
        for pt in self.current_path:
            d = math.hypot(pt[0] - self.robot_x, pt[1] - self.robot_y)
            if d >= 0.8:
                return pt
        return self.current_path[-1]

    def normalize_angle(self, a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(PathFollower())
    rclpy.shutdown()


if __name__ == '__main__':
    main()