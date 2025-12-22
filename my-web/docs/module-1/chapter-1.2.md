---
sidebar_position: 2
title: "Chapter 1.2: ROS2 Architecture & Nodes"
description: "Understanding ROS2 architecture, nodes, and communication patterns in robotics development"
---

# Chapter 1.2: ROS2 Architecture & Nodes

## Learning Objectives

By the end of this chapter, you will be able to:
- Explain the core concepts of ROS2 architecture
- Create and implement ROS2 nodes using Python
- Understand the pub/sub communication pattern
- Use services and actions in ROS2
- Implement client-server communication in robotics systems

## Introduction to ROS2 Architecture

ROS2 (Robot Operating System 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

Unlike traditional monolithic software architectures, ROS2 follows a distributed approach where computation is spread across multiple processes and potentially multiple machines. This architecture provides several advantages:
- **Modularity**: Different components can be developed and tested independently
- **Fault tolerance**: Failure of one component doesn't necessarily bring down the entire system
- **Scalability**: New components can be added without disrupting existing functionality
- **Language agnostic**: Components can be written in different programming languages

## Core Concepts of ROS2

### Nodes
A node is a process that performs computation. ROS2 is designed with the philosophy that each node should perform a single, modular function. Nodes communicate with other nodes through topics, services, and actions.

### Topics and Messages
Topics are named buses over which nodes exchange messages. A node that sends a message to a topic is called a publisher, and a node that receives messages from a topic is called a subscriber. This creates a publish/subscribe communication pattern.

Messages are data structures exchanged between nodes. They are defined using a special interface definition language (IDL) and are strongly typed.

### Services
Services provide a request/reply communication pattern. A node that provides a service is called a service server, and a node that requests the service is called a service client.

### Actions
Actions are used for long-running tasks that provide feedback and goal preemption capabilities. They follow a three-part communication pattern: goal, feedback, and result.

## Setting Up Your First ROS2 Node

Let's start by creating a simple ROS2 node in Python. This node will demonstrate the basic structure of a ROS2 node:

```python
#!/usr/bin/env python3
"""
Simple ROS2 Publisher Node

This node demonstrates the basic structure of a ROS2 node by publishing
a counter message to a topic at regular intervals.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time


class SimplePublisher(Node):
    """
    A simple ROS2 publisher node that publishes counter messages.
    """

    def __init__(self):
        # Initialize the node with a name
        super().__init__('simple_publisher')

        # Create a publisher for String messages on the 'counter' topic
        self.publisher_ = self.create_publisher(String, 'counter', 10)

        # Create a timer that calls the timer_callback function every 0.5 seconds
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Initialize a counter variable
        self.i = 0

        self.get_logger().info('Simple Publisher Node Initialized')

    def timer_callback(self):
        """Callback function called by the timer."""
        # Create a String message
        msg = String()
        msg.data = f'Hello World: {self.i}'

        # Publish the message
        self.publisher_.publish(msg)

        # Log the message
        self.get_logger().info(f'Publishing: "{msg.data}"')

        # Increment the counter
        self.i += 1


def main(args=None):
    """Main function to run the simple publisher node."""
    # Initialize ROS2
    rclpy.init(args=args)

    # Create the node
    simple_publisher = SimplePublisher()

    try:
        # Spin the node so the callback function is called
        rclpy.spin(simple_publisher)
    except KeyboardInterrupt:
        simple_publisher.get_logger().info('Shutting down Simple Publisher Node')
    finally:
        # Destroy the node explicitly
        simple_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Example 2: Simple Subscriber Node

Now let's create a subscriber node that listens to the messages published by our first node:

```python
#!/usr/bin/env python3
"""
Simple ROS2 Subscriber Node

This node demonstrates how to subscribe to messages published on a topic.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimpleSubscriber(Node):
    """
    A simple ROS2 subscriber node that listens to messages on the 'counter' topic.
    """

    def __init__(self):
        # Initialize the node with a name
        super().__init__('simple_subscriber')

        # Create a subscription to the 'counter' topic
        self.subscription = self.create_subscription(
            String,
            'counter',
            self.listener_callback,
            10)  # Queue size

        # Prevent unused variable warning
        self.subscription  # type: ignore

        self.get_logger().info('Simple Subscriber Node Initialized')

    def listener_callback(self, msg):
        """Callback function called when a message is received."""
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    """Main function to run the simple subscriber node."""
    # Initialize ROS2
    rclpy.init(args=args)

    # Create the node
    simple_subscriber = SimpleSubscriber()

    try:
        # Spin the node so the callback function is called
        rclpy.spin(simple_subscriber)
    except KeyboardInterrupt:
        simple_subscriber.get_logger().info('Shutting down Simple Subscriber Node')
    finally:
        # Destroy the node explicitly
        simple_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Example 3: Service Server and Client

Now let's implement a service-based communication pattern. First, the service server:

```python
#!/usr/bin/env python3
"""
ROS2 Service Server Example

This node implements a simple calculator service that can add two integers.
"""

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class CalculatorService(Node):
    """
    A service server that provides addition functionality.
    """

    def __init__(self):
        super().__init__('calculator_service')

        # Create a service that handles AddTwoInts requests
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback
        )

        self.get_logger().info('Calculator Service Server Initialized')

    def add_two_ints_callback(self, request, response):
        """Callback function to handle service requests."""
        # Perform the addition
        response.sum = request.a + request.b

        # Log the operation
        self.get_logger().info(
            f'Request received: {request.a} + {request.b} = {response.sum}'
        )

        # Return the response
        return response


def main(args=None):
    """Main function to run the calculator service server."""
    rclpy.init(args=args)

    calculator_service = CalculatorService()

    try:
        rclpy.spin(calculator_service)
    except KeyboardInterrupt:
        calculator_service.get_logger().info('Shutting down Calculator Service Server')
    finally:
        calculator_service.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

And now the corresponding service client:

```python
#!/usr/bin/env python3
"""
ROS2 Service Client Example

This node demonstrates how to call the calculator service to add two integers.
"""

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts
import sys


class CalculatorClient(Node):
    """
    A service client that calls the calculator service to add two integers.
    """

    def __init__(self):
        super().__init__('calculator_client')

        # Create a client for the 'add_two_ints' service
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for the service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = AddTwoInts.Request()

        self.get_logger().info('Calculator Client Initialized')

    def send_request(self, a, b):
        """Send a request to the service."""
        self.req.a = a
        self.req.b = b

        # Call the service asynchronously
        self.future = self.cli.call_async(self.req)

        return self.future


def main(args=None):
    """Main function to run the calculator client."""
    rclpy.init(args=args)

    calculator_client = CalculatorClient()

    # Check if command line arguments are provided
    if len(sys.argv) != 3:
        calculator_client.get_logger().info('Usage: ros2 run pkg_name calculator_client.py <int1> <int2>')
        calculator_client.get_logger().info('Using default values: a=2, b=3')
        a = 2
        b = 3
    else:
        a = int(sys.argv[1])
        b = int(sys.argv[2])

    # Send the request
    future = calculator_client.send_request(a, b)

    # Spin until the future is complete
    rclpy.spin_until_future_complete(calculator_client, future)

    # Process the response
    if future.result() is not None:
        response = future.result()
        calculator_client.get_logger().info(
            f'Result of {a} + {b} = {response.sum}'
        )
    else:
        calculator_client.get_logger().error(
            f'Exception while calling service: {future.exception()}'
        )

    calculator_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Example 4: Parameter Server Usage

ROS2 provides a parameter server that allows nodes to configure themselves at runtime:

```python
#!/usr/bin/env python3
"""
ROS2 Parameter Server Example

This node demonstrates how to use parameters in ROS2.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String


class ParameterDemo(Node):
    """
    A node that demonstrates parameter usage in ROS2.
    """

    def __init__(self):
        super().__init__('parameter_demo')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'turtlebot')
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('operational_mode', 'normal')

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_speed = self.get_parameter('max_speed').value
        self.operational_mode = self.get_parameter('operational_mode').value

        # Create a publisher
        self.publisher_ = self.create_publisher(String, 'robot_status', 10)

        # Create a timer to periodically publish status
        self.timer = self.create_timer(1.0, self.status_callback)

        # Add callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info(
            f'Parameter Demo Node Initialized with:\n'
            f'  Robot Name: {self.robot_name}\n'
            f'  Max Speed: {self.max_speed}\n'
            f'  Mode: {self.operational_mode}'
        )

    def parameter_callback(self, params):
        """Callback for parameter changes."""
        for param in params:
            if param.name == 'robot_name':
                self.robot_name = param.value
            elif param.name == 'max_speed':
                self.max_speed = param.value
            elif param.name == 'operational_mode':
                self.operational_mode = param.value

        self.get_logger().info(
            f'Parameters updated:\n'
            f'  Robot Name: {self.robot_name}\n'
            f'  Max Speed: {self.max_speed}\n'
            f'  Mode: {self.operational_mode}'
        )

        return rclpy.node.SetParametersResult(successful=True)

    def status_callback(self):
        """Publish robot status."""
        msg = String()
        msg.data = f'Status: {self.robot_name} in {self.operational_mode} mode, max speed: {self.max_speed}'
        self.publisher_.publish(msg)

        self.get_logger().info(f'Published: {msg.data}')


def main(args=None):
    """Main function to run the parameter demo node."""
    rclpy.init(args=args)

    parameter_demo = ParameterDemo()

    try:
        rclpy.spin(parameter_demo)
    except KeyboardInterrupt:
        parameter_demo.get_logger().info('Shutting down Parameter Demo Node')
    finally:
        parameter_demo.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Example 5: Lifecycle Nodes

Lifecycle nodes provide a way to manage the state of a node through well-defined states:

```python
#!/usr/bin/env python3
"""
ROS2 Lifecycle Node Example

This node demonstrates lifecycle management in ROS2.
"""

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState as State
from std_msgs.msg import String


class LifecycleDemo(LifecycleNode):
    """
    A lifecycle node that demonstrates state management in ROS2.
    """

    def __init__(self):
        super().__init__('lifecycle_demo')

        # Initialize publisher as None, will be created in on_activate
        self.pub = None

        self.get_logger().info('Lifecycle Demo Node Initialized')

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Called when transitioning to configuring state."""
        self.get_logger().info(f'Configuring node, previous state: {state.label}')

        # Create publisher
        self.pub = self.create_publisher(String, 'lifecycle_status', 10)

        # Create a timer that will be activated later
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.timer.cancel()  # Cancel initially, will activate later

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Called when transitioning to activating state."""
        self.get_logger().info(f'Activating node, previous state: {state.label}')

        # Activate the publisher and timer
        self.pub.on_activate()
        self.timer.reset()

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Called when transitioning to deactivating state."""
        self.get_logger().info(f'Deactivating node, previous state: {state.label}')

        # Deactivate the publisher and timer
        self.pub.on_deactivate()
        self.timer.cancel()

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """Called when transitioning to cleaningup state."""
        self.get_logger().info(f'Cleaning up node, previous state: {state.label}')

        # Destroy publisher and timer
        self.destroy_publisher(self.pub)
        self.destroy_timer(self.timer)
        self.pub = None

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """Called when transitioning to shuttingdown state."""
        self.get_logger().info(f'Shutting down node, previous state: {state.label}')

        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state: State) -> TransitionCallbackReturn:
        """Called when transitioning to errorprocessing state."""
        self.get_logger().error(f'Error in node, previous state: {state.label}')

        return TransitionCallbackReturn.FAILURE

    def timer_callback(self):
        """Callback function for the timer."""
        msg = String()
        msg.data = f'Lifecycle node is active: {self.get_current_state().label}'
        self.pub.publish(msg)

        self.get_logger().info(f'Published: {msg.data}')


def main(args=None):
    """Main function to run the lifecycle demo node."""
    rclpy.init(args=args)

    lifecycle_demo = LifecycleDemo()

    try:
        # In a real application, you'd typically use lifecycle manager
        # For demonstration, we'll manually transition states
        rclpy.spin(lifecycle_demo)
    except KeyboardInterrupt:
        lifecycle_demo.get_logger().info('Shutting down Lifecycle Demo Node')
    finally:
        lifecycle_demo.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Understanding Node Communication Patterns

### Publish/Subscribe Pattern
The publish/subscribe pattern is ideal for streaming data between nodes. Publishers send messages to topics without knowing who will receive them, and subscribers receive messages from topics without knowing who sent them. This loose coupling enables great flexibility in system design.

### Service/Client Pattern
The service/client pattern is synchronous and request-response based. It's ideal for operations that have a clear beginning and end, such as configuration requests or computational tasks.

### Action-Based Communication
Actions are designed for long-running tasks that require feedback and the ability to cancel goals. They're perfect for navigation, manipulation, or any task that takes time to complete.

## Best Practices for Node Development

1. **Keep nodes focused**: Each node should have a single responsibility
2. **Use appropriate message types**: Choose the right message type for your data
3. **Handle exceptions gracefully**: Always consider what happens when things go wrong
4. **Log appropriately**: Use different log levels (debug, info, warn, error) appropriately
5. **Manage resources**: Properly clean up publishers, subscribers, timers, etc.
6. **Use parameters**: Make your nodes configurable through parameters
7. **Consider timing**: Be mindful of timing requirements and real-time constraints

## Exercises

1. **Basic Implementation**:
   - Create a publisher node that publishes temperature readings
   - Create a subscriber node that processes these readings
   - Set up the communication and verify it works

2. **Service Extension**:
   - Modify the calculator service to handle multiple operations (add, subtract, multiply, divide)
   - Create a client that can choose which operation to perform

3. **Parameter Tuning**:
   - Add more parameters to the parameter demo node
   - Experiment with changing parameters at runtime using ROS2 command line tools

4. **Architecture Design**:
   - Design a ROS2 architecture for a mobile robot with sensors, actuators, and processing nodes
   - Identify which nodes should be created and how they should communicate

5. **Real-World Application**:
   - Choose a real-world robot application (vacuum cleaner, delivery robot, etc.)
   - Design a ROS2 node structure that would support this application

## Summary

ROS2 provides a robust framework for building distributed robotic systems. Understanding nodes, topics, services, and actions is fundamental to creating effective robotic applications. The architecture promotes modularity, fault tolerance, and scalability, making it an excellent choice for complex robotic systems.

In the next chapter, we'll explore topics and services in more detail, including advanced communication patterns and best practices for handling real-world scenarios.