#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
import heapq
import math


class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')

        # Map parameters (world frame)
        self.resolution = 0.2
        self.width_m = 15.0
        self.height_m = 20.0
        self.width_cells = int(self.width_m / self.resolution)
        self.height_cells = int(self.height_m / self.resolution)

        # Robot was spawned at world (7.5, 1.5) — odom frame is offset from world
        self.spawn_x = 7.5
        self.spawn_y = 1.5

        # Build map (in world coordinates)
        self.grid = self.build_warehouse_map()

        # Robot position in world coordinates (updated from /odom)
        self.robot_x = self.spawn_x
        self.robot_y = self.spawn_y

        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # Publisher
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

        self.get_logger().info(
            f'A* Planner ready. Map: {self.width_cells}x{self.height_cells} cells. '
            f'Spawn at world ({self.spawn_x}, {self.spawn_y})')

    # -----------------------------------------------------
    # MAP BUILDING
    # -----------------------------------------------------
    def world_to_grid(self, x, y):
        return int(x / self.resolution), int(y / self.resolution)

    def grid_to_world(self, gx, gy):
        return gx * self.resolution + self.resolution / 2, \
               gy * self.resolution + self.resolution / 2

    def add_wall(self, xc, yc, lx, ly):
        gx_min, gy_min = self.world_to_grid(xc - lx/2, yc - ly/2)
        gx_max, gy_max = self.world_to_grid(xc + lx/2, yc + ly/2)
        for gy in range(max(0, gy_min), min(self.height_cells, gy_max + 1)):
            for gx in range(max(0, gx_min), min(self.width_cells, gx_max + 1)):
                self.grid[gy][gx] = 1

    def build_warehouse_map(self):
        self.grid = [[0]*self.width_cells for _ in range(self.height_cells)]
        # Outer walls
        self.add_wall(7.5, 20.0, 15.0, 0.2)
        self.add_wall(7.5, 0.0,  15.0, 0.2)
        self.add_wall(0.0, 10.0, 0.2, 20.0)
        self.add_wall(15.0,10.0, 0.2, 20.0)
        # Aisle walls
        self.add_wall(1.0, 12.5, 0.2, 15.0)
        self.add_wall(4.0, 12.5, 0.2, 15.0)
        self.add_wall(5.0, 12.5, 0.2, 15.0)
        self.add_wall(8.0, 12.5, 0.2, 15.0)
        self.add_wall(9.0, 12.5, 0.2, 15.0)
        self.add_wall(12.0,12.5, 0.2, 15.0)
        return self.grid

    # -----------------------------------------------------
    # CALLBACKS
    # -----------------------------------------------------
    def odom_callback(self, msg):
        # /odom is relative to spawn — convert to world frame
        self.robot_x = msg.pose.pose.position.x + self.spawn_x
        self.robot_y = msg.pose.pose.position.y + self.spawn_y

    def goal_callback(self, msg):
        # Goal arrives in odom frame — convert to world frame
        goal_world_x = msg.pose.position.x + self.spawn_x
        goal_world_y = msg.pose.position.y + self.spawn_y
        self.get_logger().info(
            f'Goal received: world=({goal_world_x:.2f}, {goal_world_y:.2f})')
        self.plan_path(self.robot_x, self.robot_y, goal_world_x, goal_world_y)

    # -----------------------------------------------------
    # A* ALGORITHM
    # -----------------------------------------------------
    def heuristic(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def is_free(self, gx, gy):
        if gx < 0 or gy < 0 or gx >= self.width_cells or gy >= self.height_cells:
            return False
        return self.grid[gy][gx] == 0

    def astar(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                     (-1,-1),(-1,1),(1,-1),(1,1)]
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self.reconstruct_path(came_from, current)
            for dx, dy in neighbors:
                neighbor = (current[0]+dx, current[1]+dy)
                if not self.is_free(neighbor[0], neighbor[1]):
                    continue
                step_cost = math.sqrt(dx*dx + dy*dy)
                tentative_g = g_score[current] + step_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
        return None

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # -----------------------------------------------------
    # PATH PLANNING AND PUBLISHING
    # -----------------------------------------------------
    def plan_path(self, sx, sy, gx, gy):
        # Inputs are in world coordinates
        start = self.world_to_grid(sx, sy)
        goal = self.world_to_grid(gx, gy)
        self.get_logger().info(
            f'Planning: start grid={start} (world={sx:.1f},{sy:.1f}), '
            f'goal grid={goal} (world={gx:.1f},{gy:.1f})')

        if not self.is_free(*start):
            self.get_logger().warn('Start position is in a wall')
            return
        if not self.is_free(*goal):
            self.get_logger().warn('Goal position is in a wall')
            return

        grid_path = self.astar(start, goal)
        if grid_path is None:
            self.get_logger().warn('No path found')
            return

        self.get_logger().info(f'Path found with {len(grid_path)} cells')
        self.publish_path(grid_path)

    def publish_path(self, grid_path):
        path_msg = Path()
        path_msg.header.frame_id = 'odom'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for gx, gy in grid_path:
            wx, wy = self.grid_to_world(gx, gy)
            pose = PoseStamped()
            pose.header = path_msg.header
            # Convert back to odom frame for publishing
            pose.pose.position.x = wx - self.spawn_x
            pose.pose.position.y = wy - self.spawn_y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.get_logger().info('Path published on /planned_path')


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()