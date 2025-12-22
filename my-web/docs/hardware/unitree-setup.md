---
sidebar_position: 4
title: "Unitree Robot Setup Guide"
description: "Complete guide for setting up Unitree quadruped robots with control software, communication protocols, and development environment"
---

# Unitree Robot Setup Guide

## Learning Objectives

By the end of this guide, you will be able to:
- Configure Unitree quadruped robots for development and operation
- Install and set up Unitree's control software and SDK
- Establish communication protocols with Unitree robots
- Set up development environment for Unitree robot programming
- Deploy and test control algorithms on Unitree platforms

## Unitree Robot Overview

Unitree Robotics offers several quadruped robot platforms designed for research, education, and commercial applications:

### Platform Comparison

| Model | A1 | Go1 | B1 | B2 |
|-------|----|----|----|----|
| **Weight** | 12 kg | 14 kg | 10.2 kg | 23 kg |
| **Payload** | 3 kg | 5 kg | 3 kg | 10 kg |
| **Max Speed** | 3 m/s | 3 m/s | 2.8 m/s | 1.5 m/s |
| **Battery** | 16.56V/2.6Ah | 16.56V/2.6Ah | 16.56V/2.6Ah | 24V/12.5Ah |
| **Motors** | 12 (3 per leg) | 12 (3 per leg) | 12 (3 per leg) | 12 (3 per leg) |
| **Sensors** | IMU, Encoders | IMU, Encoders | IMU, Encoders | IMU, Encoders, LIDAR |
| **Connectivity** | WiFi, Ethernet | WiFi, Ethernet | WiFi, Ethernet | WiFi, Ethernet, 5G |

### Common Use Cases
- Research and development in legged locomotion
- Educational platforms for robotics
- Inspection and surveillance
- Entertainment and interaction
- Advanced control algorithm testing

## Prerequisites and Safety

### Safety Requirements
- Ensure adequate space for robot operation (minimum 3x3 meters)
- Keep clear of obstacles and people during testing
- Use safety leash if available for your model
- Have emergency stop procedure ready

### Required Equipment
- Unitree robot platform
- Remote controller (if included with your model)
- Power adapter for robot charging
- Development computer (Ubuntu 20.04/22.04 recommended)
- Ethernet cable for direct connection (recommended for initial setup)

## Initial Robot Setup

### 1. Physical Setup
```bash
# Before powering on, ensure:
# - All legs are properly attached
# - Battery is fully charged
# - No visible damage to the robot
# - Area is clear of obstacles

# Power on sequence:
# 1. Connect battery to robot
# 2. Press and hold power button for 3 seconds
# 3. Wait for LED indicators to show ready state
```

### 2. Network Configuration
```bash
# Connect to robot's WiFi network (for models with WiFi)
# Default SSID: Unitree-XXXX (where XXXX is robot serial number)
# Default password: 12345678

# Or use Ethernet connection (recommended for development):
# Connect Ethernet cable from development computer to robot
# Robot IP: 192.168.123.10 (fixed)
# Computer IP: 192.168.123.11 (set manually)
```

### 3. Verify Robot Status
```bash
# Check robot status using ping
ping 192.168.123.10

# For WiFi connection, use robot's WiFi IP
# Check with network scanner if needed:
sudo nmap -sn 192.168.1.0/24
```

## Development Environment Setup

### 1. Install Dependencies
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install basic development tools
sudo apt install -y build-essential cmake git curl wget vim htop

# Install Python 3.8+ and pip
sudo apt install -y python3.8 python3.8-dev python3.8-venv python3-pip

# Create virtual environment for Unitree development
python3 -m venv ~/unitree_env
source ~/unitree_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 2. Install Unitree SDK Dependencies
```bash
# Install system dependencies for Unitree SDK
sudo apt install -y \
    libeigen3-dev \
    libboost-all-dev \
    libopencv-dev \
    libyaml-cpp-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxext-dev \
    libx11-dev
```

## ROS2 Integration Setup

### 1. Install ROS2 Humble Hawksbill
```bash
# Add ROS2 GPG key
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add ROS2 repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update and install ROS2
sudo apt update
sudo apt install -y ros-humble-desktop

# Install ROS2 development tools
sudo apt install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool
```

### 2. Setup ROS2 Environment
```bash
# Add ROS2 setup to bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Unitree SDK Installation

### 1. Clone Unitree ROS2 Packages
```bash
# Create workspace
mkdir -p ~/unitree_ws/src
cd ~/unitree_ws/src

# Clone Unitree ROS2 packages (using official repositories)
git clone -b humble https://github.com/unitreerobotics/unitree_ros2.git
git clone -b humble https://github.com/unitreerobotics/unitree_legged_sdk.git
git clone -b humble https://github.com/unitreerobotics/unitree_ros_legged_control.git
git clone -b humble https://github.com/unitreerobotics/unitree_ros_legged_msgs.git

# Clone additional useful packages
git clone -b humble https://github.com/unitreerobotics/unitree_ros_manipulation.git  # if applicable to your model
```

### 2. Install Additional Dependencies
```bash
cd ~/unitree_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
```

### 3. Build Unitree Packages
```bash
cd ~/unitree_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install

# Add workspace to bashrc
echo "source ~/unitree_ws/install/setup.bash" >> ~/.bashrc
source ~/unitree_ws/install/setup.bash
```

## Communication Protocol Setup

### 1. Network Configuration
```bash
# Create network configuration script
cat << EOF > ~/unitree_network_setup.sh
#!/bin/bash

# Configure static IP for Unitree communication
sudo ip addr add 192.168.123.11/24 dev eth0 2>/dev/null || echo "Interface eth0 not found, trying enp*"
sudo ip addr add 192.168.123.11/24 dev enp0s3 2>/dev/null || echo "Interface enp0s3 not found"

# Enable IP forwarding
echo 1 | sudo tee /proc/sys/net/ipv4/ip_forward

# Add route to robot
sudo ip route add 192.168.123.10 via 192.168.123.10 dev \$(ip route | grep default | awk '{print \$5}')

echo "Network configuration complete"
EOF

chmod +x ~/unitree_network_setup.sh
```

### 2. Test Communication
```bash
# Run network setup
~/unitree_network_setup.sh

# Test communication with robot
ping 192.168.123.10

# Test specific ports used by Unitree
nc -zv 192.168.123.10 8080  # HTTP API
nc -zv 192.168.123.10 8007  # UDP communication
```

## Unitree Control Software Setup

### 1. Install Unitree Control Libraries
```bash
# Activate virtual environment
source ~/unitree_env/bin/activate

# Install Python control libraries
pip install numpy scipy matplotlib
pip install transforms3d pyquaternion
pip install requests websocket-client

# Install robotics libraries
pip install rospy ros_numpy
```

### 2. Configure Control Parameters
```bash
# Create configuration directory
mkdir -p ~/unitree_ws/src/unitree_ros_config/config

# Create default configuration file
cat << EOF > ~/unitree_ws/src/unitree_ros_config/config/default_config.yaml
# Unitree Robot Configuration
robot_model: "A1"  # Change to your robot model: A1, Go1, B1, B2
robot_ip: "192.168.123.10"
control_mode: "position_velocity_force"
frequency: 500  # Hz

# Joint limits (in radians)
joint_limits:
  hip_max: 0.802
  hip_min: -0.802
  thigh_max: 1.983
  thigh_min: -0.916
  calf_max: -0.697
  calf_min: -2.719

# Safety parameters
max_velocity: 1.0
max_force: 20.0
safety_margin: 0.1

# Communication settings
timeout: 0.1
retries: 3
EOF
```

## Basic Control Examples

### 1. Position Control Example
```python
#!/usr/bin/env python3
"""
Basic Unitree Robot Position Control Example
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import numpy as np
import time

class UnitreePositionController(Node):
    def __init__(self):
        super().__init__('unitree_position_controller')

        # Joint state publisher
        self.joint_pub = self.create_publisher(JointState, '/joint_commands', 10)

        # Robot state subscriber
        self.state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.state_callback,
            10
        )

        # Timer for control loop
        self.timer = self.create_timer(0.002, self.control_loop)  # 500Hz

        # Robot state
        self.current_positions = np.zeros(12)
        self.target_positions = np.zeros(12)

        # Initialize to zero position (standing pose)
        self.initial_pose = np.array([
            0.0, 0.8, -1.6,  # FR
            0.0, 0.8, -1.6,  # FL
            0.0, 0.8, -1.6,  # HR
            0.0, 0.8, -1.6   # HL
        ])

        self.target_positions = self.initial_pose.copy()

        self.get_logger().info('Unitree Position Controller initialized')

    def state_callback(self, msg):
        """Update current joint positions"""
        self.current_positions = np.array(msg.position)

    def control_loop(self):
        """Main control loop"""
        msg = JointState()
        msg.name = [
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'HR_hip_joint', 'HR_thigh_joint', 'HR_calf_joint',
            'HL_hip_joint', 'HL_thigh_joint', 'HL_calf_joint'
        ]
        msg.position = self.target_positions.tolist()
        msg.velocity = [0.0] * 12  # Zero velocity
        msg.effort = [0.0] * 12    # Zero effort (position control mode)

        self.joint_pub.publish(msg)

    def move_to_pose(self, pose):
        """Move robot to specified pose"""
        self.target_positions = np.array(pose)

def main(args=None):
    rclpy.init(args=args)

    controller = UnitreePositionController()

    # Move to initial standing pose
    controller.move_to_pose(controller.initial_pose)

    # Run for a few seconds
    start_time = time.time()
    while time.time() - start_time < 5.0:
        rclpy.spin_once(controller, timeout_sec=0.001)

    controller.get_logger().info('Position control example completed')

    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Create ROS2 Package for Examples
```bash
# In your workspace
cd ~/unitree_ws/src
source ~/unitree_ws/install/setup.bash

# Create package for Unitree examples
ros2 pkg create unitree_examples --build-type ament_python --dependencies rclpy sensor_msgs geometry_msgs std_msgs

# Copy the example to the package
cp ~/unitree_position_control.py ~/unitree_ws/src/unitree_examples/unitree_examples/
```

## Advanced Control Setup

### 1. Install MPC (Model Predictive Control) Libraries
```bash
# Activate virtual environment
source ~/unitree_env/bin/activate

# Install optimization libraries for MPC
pip install cvxpy
pip install scipy-optimize
pip install qpsolvers

# Install trajectory planning libraries
pip install pybind11
pip install casadi  # For advanced optimization
```

### 2. Install Gazebo Simulation (Optional)
```bash
# Add Gazebo repository
sudo curl -sSL https://get.gazebosim.org | sh

# Install Gazebo Garden
sudo apt install -y gz-harmonic

# Install ROS2 Gazebo packages
sudo apt install -y \
    ros-humble-gazebo-ros \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-plugins
```

## Unitree-Specific ROS2 Launch Files

### 1. Create Launch Directory
```bash
mkdir -p ~/unitree_ws/src/unitree_examples/launch

# Create basic launch file
cat << EOF > ~/unitree_ws/src/unitree_examples/launch/unitree_basic_control.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='unitree_examples',
            executable='position_controller',
            name='unitree_position_controller',
            output='screen',
            parameters=[
                os.path.join(
                    get_package_share_directory('unitree_examples'),
                    'config',
                    'default_config.yaml'
                )
            ]
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'robot_description':
                    # Load robot description from URDF
                    # This would be loaded from unitree's URDF files
                }
            ]
        )
    ])
EOF
```

## Testing and Validation

### 1. Test Basic Communication
```bash
# Source ROS2 and workspace
source /opt/ros/humble/setup.bash
source ~/unitree_ws/install/setup.bash

# Check available topics
ros2 topic list

# Check robot state (if robot is powered on and connected)
ros2 topic echo /joint_states --field position
```

### 2. Run Basic Control Example
```bash
# Build the workspace again after adding examples
cd ~/unitree_ws
colcon build --packages-select unitree_examples

# Source the workspace
source install/setup.bash

# Run the position control example
ros2 run unitree_examples position_controller
```

### 3. Unitree SDK Test
```bash
# Navigate to SDK directory
cd ~/unitree_ws/src/unitree_legged_sdk

# Build the C++ examples
mkdir build
cd build
cmake ..
make

# Run low-level control example (adjust for your robot model)
# This requires the robot to be in "free" mode using remote
./example_walk
```

## Troubleshooting Common Issues

### 1. Communication Issues
```bash
# Check network connectivity
ping 192.168.123.10

# Check if robot is responding on expected ports
nmap -p 8007,8080 192.168.123.10

# Reset network configuration if needed
sudo ip addr flush dev eth0
sudo ip addr flush dev enp0s3
```

### 2. Permission Issues
```bash
# Add user to dialout group for serial communication
sudo usermod -a -G dialout $USER

# Set proper permissions for network interfaces
sudo chmod 666 /dev/net/tun
```

### 3. Robot Safety Mode
```bash
# If robot is in safety mode, use remote to switch to "free" mode
# Or use the following command if in development mode:
# (This varies by robot model and requires proper safety setup)

# Check robot status via SDK
# The robot needs to be in the correct mode for control commands
```

## Safety and Operational Guidelines

### 1. Pre-Operation Checklist
- [ ] Robot battery is fully charged
- [ ] Operating area is clear of obstacles and people
- [ ] Network connection is established
- [ ] Emergency stop procedure is ready
- [ ] Robot is in safe starting position

### 2. During Operation
- Monitor robot status continuously
- Keep remote controller ready for emergency stop
- Watch for unusual noises or behaviors
- Maintain safe distance during dynamic movements

### 3. Post-Operation
- Return robot to safe position
- Power down properly
- Charge battery if needed
- Document any issues or observations

## Performance Optimization

### 1. Real-time Kernel (Optional)
```bash
# Install real-time kernel for better control performance
sudo apt install -y linux-image-rt-generic

# Configure CPU governor for performance
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 2. Network Optimization
```bash
# Optimize network settings for real-time communication
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.core.rmem_default=65536
sudo sysctl -w net.core.wmem_default=65536
```

## Next Steps

1. **Practice basic movements**: Start with simple position control commands
2. **Explore locomotion patterns**: Try different gaits and walking patterns
3. **Implement sensors**: Add IMU, camera, or LIDAR data processing
4. **Develop custom behaviors**: Create your own control algorithms
5. **Test in simulation**: Use Gazebo simulation before real robot testing

## Additional Resources

- [Unitree Official Documentation](https://www.unitree.com/docs/)
- [Unitree ROS2 GitHub Repository](https://github.com/unitreerobotics/unitree_ros2)
- [Unitree Developer Forum](https://dev.unitree.com/)
- [Quadruped Robotics Research Papers](https://arxiv.org/list/cs.RO/recent)
- [ROS2 Control Documentation](https://control.ros.org/)

## Appendix: Model-Specific Configurations

### A1 Configuration
```yaml
# A1 specific parameters
model: "A1"
joint_names: ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
              "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
              "HR_hip_joint", "HR_thigh_joint", "HR_calf_joint",
              "HL_hip_joint", "HL_thigh_joint", "HL_calf_joint"]
joint_offsets: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

### Go1 Configuration
```yaml
# Go1 specific parameters
model: "Go1"
joint_names: ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
              "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
              "HR_hip_joint", "HR_thigh_joint", "HR_calf_joint",
              "HL_hip_joint", "HL_thigh_joint", "HL_calf_joint"]
joint_offsets: [0.1, 0.05, -0.05, -0.1, 0.05, -0.05, 0.1, 0.05, -0.05, -0.1, 0.05, -0.05]
```