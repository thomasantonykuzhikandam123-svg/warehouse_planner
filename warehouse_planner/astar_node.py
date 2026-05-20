#!/usr/bin/env python3
"""
A* planner with:
- Costmap inflation (Lu et al. 2014)
- Soft cost map for wall avoidance
- Theta*-style line-of-sight post-smoothing (Nash et al. 2007)
- Continuous-coordinate densification
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path, OccupancyGrid
import heapq
import math


class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')

        self.resolution = 0.2
        self.width_m = 15.0
        self.height_m = 20.0
        self.width_cells = int(self.width_m / self.resolution)
        self.height_cells = int(self.height_m / self.resolution)

        # Spawn in corridor (Y=2.5 to 5.5 is corridor now)
        self.spawn_x = 4.0
        self.spawn_y = 4.0

        self.original_grid = None
        self.grid = self.build_warehouse_map()

        self.robot_x = self.spawn_x
        self.robot_y = self.spawn_y

        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.create_timer(1.0, self.publish_map)

        self.get_logger().info(
            f'A* ready. Map: {self.width_cells}x{self.height_cells}. '
            f'Spawn world ({self.spawn_x}, {self.spawn_y})')

    # -----------------------------------------------------
    # MAP HELPERS
    # -----------------------------------------------------
    def world_to_grid(self, x, y):
        return int(x / self.resolution), int(y / self.resolution)

    def grid_to_world(self, gx, gy):
        return (gx * self.resolution + self.resolution / 2,
                gy * self.resolution + self.resolution / 2)

    def add_wall_to(self, grid, xc, yc, lx, ly):
        gx_min, gy_min = self.world_to_grid(xc - lx / 2, yc - ly / 2)
        gx_max, gy_max = self.world_to_grid(xc + lx / 2, yc + ly / 2)
        for gy in range(max(0, gy_min), min(self.height_cells, gy_max + 1)):
            for gx in range(max(0, gx_min), min(self.width_cells, gx_max + 1)):
                grid[gy][gx] = 1

    # -----------------------------------------------------
    # WAREHOUSE LAYOUT
    # -----------------------------------------------------
    def build_warehouse_map(self):
        clean = [[0] * self.width_cells for _ in range(self.height_cells)]

        # Outer walls
        self.add_wall_to(clean, 7.5, 20.0, 15.0, 0.2)
        self.add_wall_to(clean, 7.5, 0.0, 15.0, 0.2)
        self.add_wall_to(clean, 0.0, 10.0, 0.2, 20.0)
        self.add_wall_to(clean, 15.0, 10.0, 0.2, 20.0)

        # Ground zones (shorter now - Y=0.5 to 2.5)
        self.add_wall_to(clean, 2.5, 1.5, 3, 2)      # packing area
        self.add_wall_to(clean, 5.0, 1.5, 1, 2)      # pallet area
        self.add_wall_to(clean, 12.0, 1.5, 4, 2)     # office

        # 4 storage racks (Y=5.5 to 17.5)
        self.add_wall_to(clean, 1.5, 11.5, 1, 12)
        self.add_wall_to(clean, 5.5, 11.5, 1, 12)
        self.add_wall_to(clean, 9.5, 11.5, 1, 12)
        self.add_wall_to(clean, 13.5, 11.5, 1, 12)

        # Pallets in side passes
        self.add_wall_to(clean, 0.6, 9.0, 0.8, 1.2)
        self.add_wall_to(clean, 14.4, 9.0, 0.8, 1.2)

        self.original_grid = [row[:] for row in clean]

        # Inflate for planning (not for display)
        inflated = self.inflate_obstacles(clean, radius_cells=2)
        self.cost_map = self.build_cost_map(inflated)

        return inflated

    def inflate_obstacles(self, grid, radius_cells=2):
        h = len(grid)
        w = len(grid[0])
        inflated = [[0] * w for _ in range(h)]
        for gy in range(h):
            for gx in range(w):
                if grid[gy][gx] == 1:
                    for dy in range(-radius_cells, radius_cells + 1):
                        for dx in range(-radius_cells, radius_cells + 1):
                            if dx * dx + dy * dy <= radius_cells * radius_cells:
                                ny, nx = gy + dy, gx + dx
                                if 0 <= ny < h and 0 <= nx < w:
                                    inflated[ny][nx] = 1
        return inflated

    def build_cost_map(self, grid):
        h = self.height_cells
        w = self.width_cells
        INF = 9999
        dist = [[INF] * w for _ in range(h)]
        queue = []
        for gy in range(h):
            for gx in range(w):
                if grid[gy][gx] == 1:
                    dist[gy][gx] = 0
                    queue.append((gx, gy))
        head = 0
        while head < len(queue):
            gx, gy = queue[head]
            head += 1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if dist[ny][nx] > dist[gy][gx] + 1:
                        dist[ny][nx] = dist[gy][gx] + 1
                        queue.append((nx, ny))

        cost = [[0.0] * w for _ in range(h)]
        influence_radius = 2
        max_penalty = 1.0
        for gy in range(h):
            for gx in range(w):
                d = dist[gy][gx]
                if d == 0:
                    cost[gy][gx] = float('inf')
                elif d < influence_radius:
                    cost[gy][gx] = max_penalty * math.exp(-d / 1.5)
                else:
                    cost[gy][gx] = 0.0
        return cost

    # -----------------------------------------------------
    # CALLBACKS
    # -----------------------------------------------------
    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x + self.spawn_x
        self.robot_y = msg.pose.pose.position.y + self.spawn_y

    def goal_callback(self, msg):
        gx_w = msg.pose.position.x + self.spawn_x
        gy_w = msg.pose.position.y + self.spawn_y
        self.get_logger().info(f'Goal: world=({gx_w:.2f}, {gy_w:.2f})')
        self.plan_path(self.robot_x, self.robot_y, gx_w, gy_w)

    # -----------------------------------------------------
    # A*
    # -----------------------------------------------------
    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def is_free(self, gx, gy):
        if gx < 0 or gy < 0 or gx >= self.width_cells or gy >= self.height_cells:
            return False
        return self.grid[gy][gx] == 0

    def astar(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self.reconstruct_path(came_from, current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.is_free(neighbor[0], neighbor[1]):
                    continue
                step_cost = math.sqrt(dx * dx + dy * dy)
                extra_cost = self.cost_map[neighbor[1]][neighbor[0]]
                tentative_g = g_score[current] + step_cost + extra_cost
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
    # LINE-OF-SIGHT POST-SMOOTHING (Nash et al. 2007)
    # -----------------------------------------------------
    def line_of_sight(self, gx1, gy1, gx2, gy2):
        """Bresenham line check."""
        x0, y0 = gx1, gy1
        x1, y1 = gx2, gy2
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            if not self.is_free(x, y):
                return False
            if x == x1 and y == y1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def straighten_path(self, grid_path):
        """Greedy shortcut: from each point, find the farthest reachable point."""
        if len(grid_path) < 3:
            return grid_path
        result = [grid_path[0]]
        i = 0
        while i < len(grid_path) - 1:
            j = len(grid_path) - 1
            while j > i + 1:
                if self.line_of_sight(grid_path[i][0], grid_path[i][1],
                                      grid_path[j][0], grid_path[j][1]):
                    break
                j -= 1
            result.append(grid_path[j])
            i = j
        return result

    # -----------------------------------------------------
    # PLANNING + PUBLISHING
    # -----------------------------------------------------
    def plan_path(self, sx, sy, gx, gy):
        start = self.world_to_grid(sx, sy)
        goal = self.world_to_grid(gx, gy)
        self.get_logger().info(f'Planning: start={start}, goal={goal}')

        if not self.is_free(*start):
            self.get_logger().warn('Start in obstacle')
            return
        if not self.is_free(*goal):
            self.get_logger().warn('Goal in obstacle')
            return

        grid_path = self.astar(start, goal)
        if grid_path is None:
            self.get_logger().warn('No path found')
            return

        original_len = len(grid_path)
        grid_path = self.straighten_path(grid_path)
        self.get_logger().info(
            f'Path: {original_len} cells -> {len(grid_path)} corners after smoothing')

        self.publish_path(grid_path)

    def publish_path(self, grid_path):
        path_msg = Path()
        path_msg.header.frame_id = 'odom'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        # Convert corners to world coords FIRST, then densify in continuous coords
        world_corners = [self.grid_to_world(gx, gy) for gx, gy in grid_path]
        dense_world = self.densify_world_path(world_corners, step_m=0.1)

        for wx, wy in dense_world:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wx - self.spawn_x
            pose.pose.position.y = wy - self.spawn_y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)
        self.get_logger().info(f'Path published: {len(dense_world)} continuous points')

    def densify_world_path(self, world_corners, step_m=0.1):
        """Densify in continuous coordinates — no grid snapping = truly straight lines."""
        if len(world_corners) < 2:
            return world_corners
        result = [world_corners[0]]
        for i in range(len(world_corners) - 1):
            x0, y0 = world_corners[i]
            x1, y1 = world_corners[i + 1]
            dx = x1 - x0
            dy = y1 - y0
            dist = math.sqrt(dx * dx + dy * dy)
            n_steps = max(1, int(dist / step_m))
            for s in range(1, n_steps + 1):
                t = s / n_steps
                result.append((x0 + t * dx, y0 + t * dy))
        return result

    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.frame_id = 'odom'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.resolution
        msg.info.width = self.width_cells
        msg.info.height = self.height_cells
        msg.info.origin.position.x = -self.spawn_x
        msg.info.origin.position.y = -self.spawn_y
        msg.info.origin.orientation.w = 1.0
        data = []
        for gy in range(self.height_cells):
            for gx in range(self.width_cells):
                data.append(100 if self.original_grid[gy][gx] == 1 else 0)
        msg.data = data
        self.map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()