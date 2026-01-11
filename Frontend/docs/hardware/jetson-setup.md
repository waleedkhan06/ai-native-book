---
sidebar_position: 3
title: "Jetson Setup Guide"
description: "Complete guide for setting up NVIDIA Jetson platforms for robotics applications with optimized edge computing configurations"
---

# Jetson Setup Guide

## Learning Objectives

By the end of this guide, you will be able to:
- Configure NVIDIA Jetson platforms for robotics applications
- Install and optimize robotics libraries for edge computing
- Set up development environment for AI/ML on Jetson
- Configure power management and thermal optimization
- Deploy and run robotics applications on Jetson platforms

## Jetson Platform Overview

NVIDIA Jetson platforms are purpose-built for AI and robotics at the edge. The current lineup includes:
- **Jetson Nano**: Entry-level platform for learning and prototyping
- **Jetson Xavier NX**: Mid-range platform with good performance-to-power ratio
- **Jetson AGX Orin**: High-performance platform for complex AI workloads
- **Jetson Orin Nano**: Cost-effective alternative to AGX Orin

### System Requirements by Platform

| Platform | CPU | GPU | RAM | Storage | Power | Use Case |
|----------|-----|-----|-----|---------|-------|----------|
| Jetson Nano | Quad-core ARM A57 | 128-core Maxwell | 4GB | 16GB eMMC | 5-10W | Learning, prototyping |
| Jetson Xavier NX | 6-core Carmel ARM v8.2 | 384-core Volta | 8GB | 16/32GB eMMC | 10-15W | Edge AI, robotics |
| Jetson AGX Orin | 12/24-core ARM v8.4 | 2048-core Ada Lovelace | 32/64GB | 64/128GB eMMC | 15-60W | High-performance AI |
| Jetson Orin Nano | 8-core ARM v8.4 | 1024-core Ada Lovelace | 4/8GB | 16/32GB eMMC | 7-15W | Cost-effective edge AI |

## Initial Jetson Setup

### 1. Flash Jetson OS

For Jetson AGX Orin, Xavier NX, and Orin Nano:
```bash
# Download NVIDIA SDK Manager from NVIDIA Developer website
# Follow the GUI to flash the OS to your Jetson device

# For headless setup, use Jetson Flash (command line alternative)
# sudo apt install nvidia-jetson-flash
```

For Jetson Nano:
```bash
# Download SD card image from NVIDIA Developer website
# Use balenaEtcher or similar tool to flash the image to SD card

# Insert SD card into Jetson Nano and power on
```

### 2. Initial Configuration
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Set up user account and password
# Follow on-screen prompts during first boot

# Configure network settings
# Connect to WiFi or configure Ethernet
```

### 3. System Information Check
```bash
# Check Jetson model and specifications
sudo apt install -y jetson-stats
jtop

# Check system status
cat /etc/nv_tegra_release
cat /proc/version
nvidia-smi
```

## Development Environment Setup

### 1. Install Python and Virtual Environment
```bash
# Install Python 3.8+ and pip
sudo apt update
sudo apt install -y python3.8 python3.8-dev python3.8-venv python3-pip

# Create a virtual environment for robotics development
python3 -m venv ~/jetson_robotics_env
source ~/jetson_robotics_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 2. Install Essential Development Tools
```bash
# Install build tools and utilities
sudo apt install -y build-essential cmake git curl wget vim htop

# Install system dependencies for robotics
sudo apt install -y \
    libeigen3-dev \
    libboost-all-dev \
    libopencv-dev \
    libyaml-cpp-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev
```

## ROS2 Installation on Jetson

### 1. Add ROS2 Repository
```bash
# Add ROS2 GPG key
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add ROS2 repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update package lists
sudo apt update
```

### 2. Install ROS2 Humble Hawksbill
```bash
# Install ROS2 core packages (minimal for Jetson)
sudo apt install -y ros-humble-ros-core

# Install essential ROS2 packages for robotics
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-tf2-tools \
    ros-humble-tf2-geometry-msgs \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-robot-localization \
    ros-humble-interactive-markers \
    ros-humble-rviz2 \
    ros-humble-vision-opencv \
    ros-humble-image-transport \
    ros-humble-compressed-image-transport \
    ros-humble-camera-info-manager \
    ros-humble-image-proc \
    ros-humble-depth-image-proc
```

### 3. Setup ROS2 Environment
```bash
# Add ROS2 setup to bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Install ROS2 development tools
sudo apt install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool
```

### 4. Initialize rosdep
```bash
# Initialize rosdep
sudo rosdep init
rosdep update
```

## AI/ML Framework Setup

### 1. Install PyTorch for Jetson
```bash
# Activate virtual environment
source ~/jetson_robotics_env/bin/activate

# Install PyTorch optimized for Jetson (check NVIDIA's PyTorch wheel page for latest)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Tensor cores available:', torch.cuda.get_device_name(0))"
```

### 2. Install TensorRT for Optimized Inference
```bash
# TensorRT is pre-installed on Jetson, but install Python bindings
pip install nvidia-tensorrt

# Install ONNX for model conversion
pip install onnx onnxruntime
```

### 3. Install Vision and Robotics Libraries
```bash
# Install computer vision libraries
pip install opencv-python-headless  # Use headless version for Jetson

# Install robotics-specific libraries
pip install transforms3d pyquaternion
pip install open3d  # May need to compile from source on Jetson

# Install AI libraries
pip install transformers accelerate
pip install supervision  # For computer vision applications
```

## Jetson-Specific Optimizations

### 1. Power Mode Configuration
```bash
# Check current power mode
sudo nvpmodel -q

# Set to maximum performance mode (for development)
sudo nvpmodel -m 0

# For production with power constraints, use:
# sudo nvpmodel -m 1  # For Jetson AGX Orin
# sudo nvpmodel -m 2  # For lower power modes
```

### 2. Jetson Clocks for Performance
```bash
# Enable maximum clocks for development
sudo jetson_clocks

# Check clock status
cat /sys/kernel/debug/clk/clk_summary | grep -E "(gpu|cpu|emc)"
```

### 3. Thermal Management
```bash
# Install jetson-stats for monitoring
sudo -H pip install -U jetson-stats
sudo jtop

# Configure fan control (if using carrier board with fan)
# Create custom thermal configuration in /etc/thermal-conf.xml
```

## Camera and Sensor Setup

### 1. Configure CSI Camera
```bash
# Test CSI camera (Jetson Nano has specific camera setup)
# For Jetson Nano, use the following:
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! nvvidconv ! 'video/x-raw,width=1920,height=1080,format=I420' ! videoconvert ! 'video/x-raw,format=BGR' ! appsink

# For other Jetson models, adapt sensor-id as needed
```

### 2. Install Camera Libraries
```bash
# Install GStreamer plugins
sudo apt install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav

# Install camera utilities
sudo apt install -y v4l-utils

# Test USB camera
v4l2-ctl --list-devices
```

## Robotics-Specific Configurations

### 1. Serial Communication Setup
```bash
# Add user to dialout group for serial communication
sudo usermod -a -G dialout $USER

# Configure serial ports
# Jetson Nano: /dev/ttyTHS1 (UART), /dev/ttyACM0 (USB serial)
# Jetson Xavier NX/AGX Orin: Multiple UART options available

# Test serial communication
ls -la /dev/tty*
```

### 2. GPIO Setup for Hardware Control
```bash
# Install Jetson.GPIO library
pip install Jetson.GPIO

# Example GPIO usage in Python
cat << EOF > test_gpio.py
import Jetson.GPIO as GPIO
import time

# Pin Definitions
led_pin = 18  # BOARD pin 12, BCM pin 18

# Pin Setup
GPIO.setmode(GPIO.BCM)  # BOARD pin-numbering scheme
GPIO.setup(led_pin, GPIO.OUT)  # Set pin as an output pin

try:
    while True:
        GPIO.output(led_pin, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(led_pin, GPIO.LOW)
        time.sleep(0.5)
finally:
    GPIO.cleanup()
EOF

python3 test_gpio.py
```

### 3. PWM Setup for Motor Control
```bash
# For hardware PWM, use Jetson PWM
# Install dependencies
sudo apt install -y python3-pip
pip install Adafruit-Blinka

# Note: Jetson has limited hardware PWM pins
# For more PWM channels, consider using PCA9685 I2C PWM driver
```

## Performance Monitoring and Optimization

### 1. Install Monitoring Tools
```bash
# Install jetson-stats for comprehensive monitoring
sudo -H pip install -U jetson-stats
sudo systemctl restart jetson_stats

# Install system monitoring
sudo apt install -y htop iotop nethogs
```

### 2. Memory Management
```bash
# Check current memory usage
free -h
cat /proc/meminfo

# For memory-constrained applications, consider swap
# Create swap file if needed (use with caution on eMMC)
sudo fallocate -l 2G /jetson_swap
sudo chmod 600 /jetson_swap
sudo mkswap /jetson_swap
sudo swapon /jetson_swap
```

### 3. Optimized Inference Setup
```bash
# Install DeepStream SDK for optimized video analytics
# Download from NVIDIA Developer website
# sudo apt install deepstream-6.3

# Install TensorRT optimization tools
pip install polygraphy
```

## ROS2 Package Development for Jetson

### 1. Create Workspace
```bash
# Create ROS2 workspace
mkdir -p ~/jetson_ws/src
cd ~/jetson_ws

# Source ROS2 and build workspace
source /opt/ros/humble/setup.bash
colcon build --symlink-install

# Add to bashrc
echo "source ~/jetson_ws/install/setup.bash" >> ~/.bashrc
```

### 2. Install Jetson-Specific ROS2 Packages
```bash
cd ~/jetson_ws/src

# Clone hardware interface packages
git clone -b humble https://github.com/ros-perception/vision_opencv.git
git clone -b humble https://github.com/ros-controls/ros2_control.git
git clone -b humble https://github.com/ros-controls/ros2_controllers.git

# Build packages
cd ~/jetson_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select vision_opencv
colcon build --packages-select ros2_control ros2_controllers
```

## Deployment and Optimization

### 1. Container Setup (Optional)
```bash
# Install Docker
sudo apt install -y docker.io
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit for GPU access
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Application Optimization
```bash
# For AI applications, use TensorRT optimization
cat << EOF > optimize_model.py
import torch
import tensorrt as trt
import numpy as np

def optimize_with_tensorrt(torch_model, input_shape):
    """
    Optimize PyTorch model with TensorRT
    """
    # Convert to ONNX first
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(torch_model, dummy_input, "model.onnx",
                      input_names=["input"], output_names=["output"])

    # Use TensorRT for optimization
    # Implementation details would depend on your specific model
    pass

print("Model optimization function ready")
EOF

python3 optimize_model.py
```

## Troubleshooting Common Issues

### 1. Memory Issues
```bash
# Check memory usage
free -h

# If running out of memory, clear cache
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Monitor memory usage during application
htop
```

### 2. GPU Issues
```bash
# Check GPU status
nvidia-smi

# If GPU not detected, check power mode
sudo nvpmodel -q
sudo jetson_clocks

# Check CUDA installation
python3 -c "import torch; print(torch.cuda.is_available())"
```

### 3. Thermal Issues
```bash
# Monitor temperature
sudo tegrastats  # Available on Jetson devices

# Or use jetson-stats
jtop

# If thermal throttling occurs, reduce power mode or improve cooling
sudo nvpmodel -m 1  # Reduce performance for lower power/heat
```

## Best Practices for Jetson Robotics

### 1. Power Management
- Use appropriate power mode for your application
- Implement power-aware algorithms
- Monitor thermal conditions continuously

### 2. Memory Management
- Use efficient data structures
- Implement memory pooling where possible
- Monitor memory usage during development

### 3. Performance Optimization
- Use TensorRT for inference when possible
- Optimize network architectures for edge deployment
- Profile applications to identify bottlenecks

### 4. Reliability
- Implement proper error handling
- Use watchdog timers for system recovery
- Plan for graceful degradation under stress

## Testing Your Setup

### 1. Test ROS2 Installation
```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Test basic ROS2 functionality
ros2 run demo_nodes_cpp talker &
ros2 run demo_nodes_cpp listener
```

### 2. Test AI Framework
```bash
# Activate virtual environment
source ~/jetson_robotics_env/bin/activate

# Test PyTorch with GPU
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    # Test tensor operations
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print('GPU tensor multiplication successful')
"
```

### 3. Test Computer Vision
```bash
# Test OpenCV installation
python3 -c "
import cv2
print('OpenCV version:', cv2.__version__)
# Test basic image operations
import numpy as np
img = np.zeros((100, 100, 3), dtype=np.uint8)
print('OpenCV basic operations work')
"
```

## Next Steps

1. **Develop your first Jetson robotics application**: Create a simple ROS2 node that uses sensors and performs basic AI inference
2. **Optimize for your specific use case**: Profile and optimize your application for performance and power consumption
3. **Deploy to hardware**: Test your application on actual robotic hardware
4. **Scale your solution**: Consider multiple Jetson platforms for distributed robotics applications

## Additional Resources

- [NVIDIA Jetson Documentation](https://docs.nvidia.com/jetson/)
- [ROS2 on Jetson Tutorial](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
- [Jetson Inference GitHub](https://github.com/dusty-nv/jetson-inference)
- [Jetson Hacks](https://www.jetsonhacks.com/) - Tutorials and examples
- [NVIDIA Developer Zone](https://developer.nvidia.com/embedded/community/jetson-projects) - Community projects