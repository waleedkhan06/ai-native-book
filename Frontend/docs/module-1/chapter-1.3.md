---
sidebar_position: 3
title: "Chapter 1.3: Topics & Services"
description: "Understanding ROS2 topics and services for inter-node communication in robotics"
---

# Chapter 1.3: Topics & Services

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the publish/subscribe communication pattern in ROS2
- Implement topics for streaming data between nodes
- Create and use services for request/reply communication
- Compare when to use topics versus services
- Implement advanced communication patterns using ROS2 interfaces

## Introduction to ROS2 Communication Patterns

Communication is the backbone of any distributed robotic system. In ROS2, nodes communicate with each other through three primary mechanisms:
- **Topics**: for asynchronous publish/subscribe communication
- **Services**: for synchronous request/reply communication
- **Actions**: for goal-oriented communication with feedback

In this chapter, we'll focus on the first two: topics and services, which form the foundation of most robotic communication patterns.

## Understanding Topics and Publishers/Subscribers

Topics in ROS2 implement a publish/subscribe communication pattern where publishers send messages to named topics and subscribers receive messages from those topics. This pattern is asynchronous and decoupled, meaning publishers and subscribers don't need to know about each other or be running simultaneously.

### Key Characteristics of Topics:
- **Asynchronous**: Publishers and subscribers operate independently
- **Many-to-many**: Multiple publishers can publish to the same topic, and multiple subscribers can listen to the same topic
- **Loose coupling**: Publishers don't know who subscribes to their topics
- **Real-time friendly**: Designed for streaming data with low latency

### Topic Example 1: Temperature Sensor Publisher

Let's create a publisher that simulates a temperature sensor:

```python
#!/usr/bin/env python3
"""
Temperature Sensor Publisher

This node simulates a temperature sensor that publishes temperature readings
to a topic at regular intervals.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Temperature
import random
import time


class TemperatureSensor(Node):
    """
    A node that simulates a temperature sensor.
    """

    def __init__(self):
        super().__init__('temperature_sensor')

        # Create a publisher for temperature messages
        self.publisher_ = self.create_publisher(Temperature, 'temperature', 10)

        # Create a timer that publishes at 1Hz
        self.timer = self.create_timer(1.0, self.timer_callback)

        # Simulate sensor drift
        self.base_temp = 25.0  # Base temperature in Celsius
        self.get_logger().info('Temperature Sensor Node Initialized')

    def timer_callback(self):
        """Publish temperature readings."""
        msg = Temperature()

        # Simulate realistic temperature readings with some noise
        current_temp = self.base_temp + random.uniform(-0.5, 0.5)

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'sensor_frame'
        msg.temperature = current_temp
        msg.variance = 0.01  # Low variance indicates reliable sensor

        self.publisher_.publish(msg)
        self.get_logger().info(f'Temperature: {msg.temperature:.2f}°C')


def main(args=None):
    """Main function to run the temperature sensor node."""
    rclpy.init(args=args)

    temp_sensor = TemperatureSensor()

    try:
        rclpy.spin(temp_sensor)
    except KeyboardInterrupt:
        temp_sensor.get_logger().info('Shutting down Temperature Sensor Node')
    finally:
        temp_sensor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Topic Example 2: Temperature Monitor Subscriber

Now let's create a subscriber that monitors the temperature readings:

```python
#!/usr/bin/env python3
"""
Temperature Monitor Subscriber

This node subscribes to temperature readings and alerts when temperatures
are outside safe ranges.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Temperature


class TemperatureMonitor(Node):
    """
    A node that monitors temperature readings and alerts if unsafe.
    """

    def __init__(self):
        super().__init__('temperature_monitor')

        # Create subscription to temperature topic
        self.subscription = self.create_subscription(
            Temperature,
            'temperature',
            self.temp_callback,
            10
        )

        # Temperature thresholds
        self.min_safe_temp = 15.0
        self.max_safe_temp = 35.0

        self.get_logger().info('Temperature Monitor Node Initialized')

    def temp_callback(self, msg):
        """Process temperature readings."""
        temp = msg.temperature

        # Check if temperature is in safe range
        if temp < self.min_safe_temp or temp > self.max_safe_temp:
            if temp < self.min_safe_temp:
                self.get_logger().warn(f'LOW TEMPERATURE ALERT: {temp:.2f}°C (below {self.min_safe_temp}°C)')
            else:
                self.get_logger().warn(f'HIGH TEMPERATURE ALERT: {temp:.2f}°C (above {self.max_safe_temp}°C)')
        else:
            self.get_logger().info(f'Normal temperature: {temp:.2f}°C')


def main(args=None):
    """Main function to run the temperature monitor node."""
    rclpy.init(args=args)

    temp_monitor = TemperatureMonitor()

    try:
        rclpy.spin(temp_monitor)
    except KeyboardInterrupt:
        temp_monitor.get_logger().info('Shutting down Temperature Monitor Node')
    finally:
        temp_monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Topic Example 3: Multiple Subscribers Pattern

Let's create another subscriber that logs temperature data for historical analysis:

```python
#!/usr/bin/env python3
"""
Temperature Logger

This node subscribes to temperature readings and logs them for historical analysis.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Temperature
from datetime import datetime


class TemperatureLogger(Node):
    """
    A node that logs temperature readings for historical analysis.
    """

    def __init__(self):
        super().__init__('temperature_logger')

        # Create subscription to temperature topic
        self.subscription = self.create_subscription(
            Temperature,
            'temperature',
            self.temp_callback,
            10
        )

        # Initialize log
        self.readings_log = []
        self.log_interval = 10  # Log summary every N readings

        self.get_logger().info('Temperature Logger Node Initialized')

    def temp_callback(self, msg):
        """Log temperature readings."""
        temp = msg.temperature
        timestamp = datetime.fromtimestamp(msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)

        # Store reading
        reading = {'timestamp': timestamp, 'temperature': temp}
        self.readings_log.append(reading)

        # Log every 10 readings
        if len(self.readings_log) % self.log_interval == 0:
            self.log_summary()

    def log_summary(self):
        """Log a summary of recent readings."""
        recent_readings = self.readings_log[-self.log_interval:]
        temps = [r['temperature'] for r in recent_readings]

        avg_temp = sum(temps) / len(temps)
        min_temp = min(temps)
        max_temp = max(temps)

        self.get_logger().info(
            f'Temperature Log Summary (last {self.log_interval} readings):\n'
            f'  Average: {avg_temp:.2f}°C\n'
            f'  Min: {min_temp:.2f}°C\n'
            f'  Max: {max_temp:.2f}°C'
        )


def main(args=None):
    """Main function to run the temperature logger node."""
    rclpy.init(args=args)

    temp_logger = TemperatureLogger()

    try:
        rclpy.spin(temp_logger)
    except KeyboardInterrupt:
        temp_logger.get_logger().info('Shutting down Temperature Logger Node')
    finally:
        temp_logger.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Understanding Services and Clients

Services in ROS2 implement a synchronous request/reply communication pattern. A service client sends a request to a service server, and the server processes the request and returns a response. This pattern is ideal for operations that have a clear beginning and end.

### Key Characteristics of Services:
- **Synchronous**: Client waits for response from server
- **One-to-one**: Each request goes to one server, receives one response
- **Request-response**: Well-defined inputs and outputs
- **Stateless**: Each request is independent of others

### Service Example 1: Navigation Service Server

Let's create a navigation service that calculates paths:

```python
#!/usr/bin/env python3
"""
Navigation Service Server

This node provides a navigation service that calculates paths between waypoints.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from nav_msgs.srv import GetPlan
import math


class NavigationService(Node):
    """
    A service server that provides navigation planning functionality.
    """

    def __init__(self):
        super().__init__('navigation_service')

        # Create the service
        self.srv = self.create_service(
            GetPlan,
            'get_navigation_plan',
            self.nav_plan_callback
        )

        # Simulate a map with obstacles
        self.obstacles = [
            Point(x=1.0, y=1.0, z=0.0),
            Point(x=2.0, y=2.0, z=0.0),
            Point(x=3.0, y=1.0, z=0.0)
        ]

        self.get_logger().info('Navigation Service Server Initialized')

    def nav_plan_callback(self, request, response):
        """Calculate a navigation plan from start to goal."""
        start = request.start.pose.position
        goal = request.goal.pose.position

        self.get_logger().info(
            f'Received navigation request from ({start.x:.2f}, {start.y:.2f}) to ({goal.x:.2f}, {goal.y:.2f})'
        )

        # Simple path calculation (in a real system, this would be more sophisticated)
        path_points = self.calculate_simple_path(start, goal)

        # Create response
        response.plan.header.stamp = self.get_clock().now().to_msg()
        response.plan.header.frame_id = 'map'
        response.plan.poses = []

        # Convert path points to poses
        for point in path_points:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = 'map'
            pose_stamped.pose.position = point
            pose_stamped.pose.orientation.w = 1.0  # No rotation
            response.plan.poses.append(pose_stamped)

        # Simulate success/failure based on obstacle proximity
        if self.is_path_clear(start, goal):
            response.plan.header.frame_id = 'map'
            self.get_logger().info(f'Calculated path with {len(path_points)} waypoints')
        else:
            self.get_logger().warn('Path contains obstacles!')

        return response

    def calculate_simple_path(self, start, goal):
        """Calculate a simple straight-line path."""
        # Calculate distance
        dx = goal.x - start.x
        dy = goal.y - start.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Sample points along the path
        num_waypoints = max(2, int(distance * 2))  # At least 2 points, ~0.5m spacing
        points = []

        for i in range(num_waypoints):
            t = i / (num_waypoints - 1) if num_waypoints > 1 else 0
            point = Point()
            point.x = start.x + t * dx
            point.y = start.y + t * dy
            point.z = 0.0
            points.append(point)

        return points

    def is_path_clear(self, start, goal):
        """Check if path is clear of obstacles."""
        # Simple collision detection
        for obs in self.obstacles:
            # Check if obstacle is near the path
            dist_to_start = math.sqrt((obs.x - start.x)**2 + (obs.y - start.y)**2)
            dist_to_goal = math.sqrt((obs.x - goal.x)**2 + (obs.y - goal.y)**2)
            path_dist = math.sqrt((goal.x - start.x)**2 + (goal.y - start.y)**2)

            # If obstacle is close to the path segment
            if dist_to_start + dist_to_goal < path_dist + 1.0:
                return False
        return True


def main(args=None):
    """Main function to run the navigation service server."""
    # Import PoseStamped here to avoid circular import issues
    from geometry_msgs.msg import PoseStamped

    rclpy.init(args=args)

    nav_service = NavigationService()

    try:
        rclpy.spin(nav_service)
    except KeyboardInterrupt:
        nav_service.get_logger().info('Shutting down Navigation Service Server')
    finally:
        nav_service.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Service Example 2: Navigation Service Client

Now let's create a client for the navigation service:

```python
#!/usr/bin/env python3
"""
Navigation Service Client

This node demonstrates how to call the navigation service to get path plans.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, Quaternion
from nav_msgs.srv import GetPlan
from nav_msgs.msg import Path
import sys


class NavigationClient(Node):
    """
    A service client that requests navigation plans from the navigation service.
    """

    def __init__(self):
        super().__init__('navigation_client')

        # Create the client
        self.cli = self.create_client(GetPlan, 'get_navigation_plan')

        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Navigation service not available, waiting...')

        self.request = GetPlan.Request()
        self.get_logger().info('Navigation Client Initialized')

    def send_nav_request(self, start_x, start_y, goal_x, goal_y):
        """Send a navigation request."""
        # Set start position
        self.request.start.header.stamp = self.get_clock().now().to_msg()
        self.request.start.header.frame_id = 'map'
        self.request.start.pose.position.x = start_x
        self.request.start.pose.position.y = start_y
        self.request.start.pose.position.z = 0.0
        self.request.start.pose.orientation = Quaternion(w=1.0)

        # Set goal position
        self.request.goal.header.stamp = self.get_clock().now().to_msg()
        self.request.goal.header.frame_id = 'map'
        self.request.goal.pose.position.x = goal_x
        self.request.goal.pose.position.y = goal_y
        self.request.goal.pose.position.z = 0.0
        self.request.goal.pose.orientation = Quaternion(w=1.0)

        # Make asynchronous call
        self.future = self.cli.call_async(self.request)
        return self.future


def main(args=None):
    """Main function to run the navigation client."""
    rclpy.init(args=args)

    nav_client = NavigationClient()

    # Parse command line arguments or use defaults
    if len(sys.argv) != 5:
        nav_client.get_logger().info(
            'Usage: ros2 run pkg_name nav_client.py <start_x> <start_y> <goal_x> <goal_y>'
        )
        nav_client.get_logger().info('Using default values: start=(0,0), goal=(5,5)')
        start_x, start_y = 0.0, 0.0
        goal_x, goal_y = 5.0, 5.0
    else:
        start_x, start_y = float(sys.argv[1]), float(sys.argv[2])
        goal_x, goal_y = float(sys.argv[3]), float(sys.argv[4])

    # Send the navigation request
    future = nav_client.send_nav_request(start_x, start_y, goal_x, goal_y)

    # Wait for the response
    try:
        rclpy.spin_until_future_complete(nav_client, future)
    except KeyboardInterrupt:
        nav_client.get_logger().info('Interrupted, shutting down')
        return

    # Process the response
    if future.result() is not None:
        response = future.result()
        if len(response.plan.poses) > 0:
            num_waypoints = len(response.plan.poses)
            nav_client.get_logger().info(
                f'Navigation plan received with {num_waypoints} waypoints'
            )
            # Print first and last waypoints
            if num_waypoints >= 2:
                start_pose = response.plan.poses[0].pose.position
                end_pose = response.plan.poses[-1].pose.position
                nav_client.get_logger().info(
                    f'  Start: ({start_pose.x:.2f}, {start_pose.y:.2f})\n'
                    f'  Goal:  ({end_pose.x:.2f}, {end_pose.y:.2f})'
                )
        else:
            nav_client.get_logger().warn('Received empty navigation plan')
    else:
        nav_client.get_logger().error(f'Service call failed: {future.exception()}')

    nav_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Comparing Topics vs Services

| Aspect | Topics | Services |
|--------|--------|----------|
| **Communication Pattern** | Publish/Subscribe | Request/Reply |
| **Synchronization** | Asynchronous | Synchronous |
| **Timing** | Continuous streaming | On-demand |
| **Coupling** | Loose | Tighter |
| **Use Case** | Sensor data, status updates | Configuration, computations |
| **Reliability** | Best-effort | Guaranteed delivery |

## Advanced Topic Features

### Quality of Service (QoS) Settings

ROS2 provides Quality of Service settings to fine-tune communication behavior:

```python
#!/usr/bin/env python3
"""
QoS Configuration Example

This node demonstrates different QoS profiles for various communication needs.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


class QoSExample(Node):
    """
    A node demonstrating different QoS configurations.
    """

    def __init__(self):
        super().__init__('qos_example')

        # Reliable communication - all messages guaranteed to arrive
        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )
        self.reliable_pub = self.create_publisher(String, 'reliable_topic', reliable_qos)

        # Best-effort communication - faster but not all messages guaranteed
        best_effort_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        self.best_effort_pub = self.create_publisher(String, 'best_effort_topic', best_effort_qos)

        # Keep-all history - retains all messages (use carefully!)
        keep_all_qos = QoSProfile(
            depth=0,  # 0 means keep all
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.keep_all_pub = self.create_publisher(String, 'keep_all_topic', keep_all_qos)

        # Timer to send messages
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.counter = 0

        self.get_logger().info('QoS Example Node Initialized')

    def timer_callback(self):
        """Send messages with different QoS settings."""
        msg = String()
        msg.data = f'Message {self.counter}'

        self.reliable_pub.publish(msg)
        self.best_effort_pub.publish(msg)
        self.keep_all_pub.publish(msg)

        self.get_logger().info(f'Sent: "{msg.data}" to all topics')
        self.counter += 1


def main(args=None):
    """Main function to run the QoS example node."""
    rclpy.init(args=args)

    qos_example = QoSExample()

    try:
        rclpy.spin(qos_example)
    except KeyboardInterrupt:
        qos_example.get_logger().info('Shutting down QoS Example Node')
    finally:
        qos_example.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Best Practices for Topics and Services

### Topic Best Practices:
1. **Choose appropriate message types**: Use standard message types when possible
2. **Set appropriate queue sizes**: Balance memory usage with message loss tolerance
3. **Use QoS settings wisely**: Match QoS to your application's requirements
4. **Avoid heavy messages**: Break down large data into smaller, more frequent messages
5. **Include timestamps**: Use header timestamps for synchronization

### Service Best Practices:
1. **Keep services stateless**: Each request should be independent
2. **Set reasonable timeouts**: Don't wait indefinitely for responses
3. **Handle errors gracefully**: Return appropriate error codes
4. **Don't use for streaming data**: Use topics instead of repeated service calls
5. **Design clear interfaces**: Make service requests and responses intuitive

## Exercises

1. **Topic Implementation**:
   - Create a publisher that streams sensor data (e.g., IMU readings)
   - Create multiple subscribers that process this data differently
   - Experiment with different QoS settings and observe the effects

2. **Service Implementation**:
   - Create a service that performs mathematical calculations
   - Implement a client that sends multiple requests and handles responses
   - Add error handling for invalid requests

3. **Integration Challenge**:
   - Combine topics and services in a single application
   - For example, use a service to configure a node, then use topics to stream its output

4. **Performance Analysis**:
   - Measure the latency and throughput of your topic-based communication
   - Compare the response times of different service configurations

5. **Real-World Scenario**:
   - Design a communication architecture for a specific robot application
   - Identify which parts should use topics vs. services

## Summary

Topics and services form the foundation of ROS2 communication. Topics provide asynchronous, decoupled communication ideal for streaming data and status updates. Services provide synchronous, request-response communication perfect for configuration and computational tasks.

Understanding when to use each communication pattern and how to configure them appropriately is crucial for building robust and efficient robotic systems. The flexibility of ROS2's communication system allows you to create sophisticated distributed applications that can handle the complex requirements of modern robotics.

In the next chapter, we'll explore URDF (Unified Robot Description Format) and how to model robots for simulation and control.