---
sidebar_position: 2
title: "Workstation Setup Guide"
description: "Complete guide for setting up a robotics development workstation with all required tools and libraries"
---

# Workstation Setup Guide

## Learning Objectives

By the end of this guide, you will be able to:
- Configure a complete robotics development environment
- Install and set up essential tools and libraries
- Set up simulation environments for robotics development
- Configure development tools for efficient workflow
- Troubleshoot common setup issues

## System Requirements

### Minimum Requirements
- **CPU**: Intel i5 or AMD Ryzen 5 (4 cores, 8 threads)
- **RAM**: 16 GB DDR4
- **Storage**: 500 GB SSD
- **GPU**: NVIDIA GTX 1060 6GB or equivalent (for AI/ML tasks)
- **OS**: Ubuntu 20.04 LTS or Ubuntu 22.04 LTS

### Recommended Requirements
- **CPU**: Intel i7/i9 or AMD Ryzen 7/9 (8+ cores, 16+ threads)
- **RAM**: 32 GB DDR4
- **Storage**: 1 TB+ NVMe SSD
- **GPU**: NVIDIA RTX 3070/3080/4070 or equivalent (for AI/ML tasks)
- **OS**: Ubuntu 22.04 LTS

## Initial System Setup

### 1. Update System Packages
```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install basic development tools
sudo apt install -y build-essential cmake git curl wget vim htop
```

### 2. Install Python and Virtual Environment
```bash
# Install Python 3.10+ and pip
sudo apt install -y python3.10 python3.10-dev python3.10-venv python3-pip

# Create a virtual environment for robotics development
python3 -m venv ~/robotics_env
source ~/robotics_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Install Essential Development Libraries
```bash
# Install system dependencies for robotics development
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
    libx11-dev \
    libxmu-dev \
    libxft-dev \
    libxss-dev \
    libxt-dev \
    libasound2-dev \
    libpulse-dev \
    libsndfile1-dev \
    libportaudio2 \
    portaudio19-dev
```

## ROS2 Installation

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
# Install ROS2 desktop package
sudo apt install -y ros-humble-desktop

# Install additional ROS2 packages
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-tf2-tools \
    ros-humble-tf2-geometry-msgs \
    ros-humble-tf2-eigen \
    ros-humble-tf2-ros \
    ros-humble-tf2-sensor-msgs \
    ros-humble-tf2-kdl \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-dwb-core \
    ros-humble-dwb-plugins \
    ros-humble-robot-localization \
    ros-humble-interactive-markers \
    ros-humble-rviz2 \
    ros-humble-rosbridge-suite \
    ros-humble-rosbridge-server \
    ros-humble-rosbridge-websocket \
    ros-humble-rosbridge-library \
    ros-humble-web-video-server \
    ros-humble-vision-opencv \
    ros-humble-image-transport \
    ros-humble-compressed-image-transport \
    ros-humble-compressed-depth-image-transport \
    ros-humble-camera-info-manager \
    ros-humble-image-proc \
    ros-humble-image-view \
    ros-humble-depth-image-proc \
    ros-humble-image-publisher \
    ros-humble-image-rotate \
    ros-humble-theora-image-transport \
    ros-humble-vision-msgs
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

## NVIDIA GPU Setup (For AI/ML Tasks)

### 1. Install NVIDIA Drivers
```bash
# Add graphics drivers PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install NVIDIA driver (adjust version as needed)
sudo apt install -y nvidia-driver-535

# Reboot to apply changes
sudo reboot
```

### 2. Install CUDA Toolkit
```bash
# Download and install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
sudo sh cuda_12.3.0_545.23.06_linux.run

# Add CUDA to environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. Install cuDNN
```bash
# Download cuDNN from NVIDIA Developer website (requires account)
# Then install:
sudo dpkg -i libcudnn8_*.deb
sudo dpkg -i libcudnn8-dev_*.deb
```

## Python Development Environment

### 1. Install Robotics Libraries
```bash
# Activate virtual environment
source ~/robotics_env/bin/activate

# Install core robotics libraries
pip install numpy scipy matplotlib pandas

# Install computer vision libraries
pip install opencv-python opencv-contrib-python

# Install deep learning frameworks
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow

# Install robotics-specific libraries
pip install pybullet transforms3d trimesh open3d

# Install ROS2 Python packages
pip install ros-numpy
pip install transforms3d
pip install quaternion
```

### 2. Install Vision-Language-Action Libraries
```bash
# Install transformers and related libraries
pip install transformers
pip install tokenizers
pip install datasets
pip install accelerate
pip install bitsandbytes

# Install robotics-specific AI libraries
pip install robomimic
pip install libero-robosuite
pip install habitat-sim
pip install habitat-lab
```

## Simulation Environment Setup

### 1. Install Isaac Sim (Optional)
```bash
# Install Isaac Sim prerequisites
sudo apt install -y \
    nvidia-prime \
    mesa-utils \
    vainfo \
    glmark2

# Download Isaac Sim from NVIDIA Developer website
# Extract and install according to NVIDIA's instructions
```

### 2. Install Gazebo Garden
```bash
# Add Gazebo repository
sudo curl -sSL https://get.gazebosim.org | sh

# Install Gazebo Garden
sudo apt install -y gz-harmonic
```

### 3. Install PyBullet for Physics Simulation
```bash
# In your virtual environment
pip install pybullet
```

## Development Tools Setup

### 1. Install Visual Studio Code
```bash
# Download VS Code
wget -O code.deb https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64
sudo dpkg -i code.deb
sudo apt install -f  # Fix any dependency issues
```

### 2. Install VS Code Extensions for Robotics
```bash
# Install essential extensions
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-vscode.cpptools
code --install-extension ms-iot.vscode-ros
code --install-extension eamodio.gitlens
code --install-extension formulahendry.ros
code --install-extension ms-python.black-formatter
code --install-extension ms-toolsai.tensorboard
```

### 3. Install Git and Configure
```bash
# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global core.editor "vim"
git config --global pull.rebase false

# Install Git Large File Storage for handling large models/datasets
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

## Workspace Setup

### 1. Create ROS2 Workspace
```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Source ROS2 and build workspace
source /opt/ros/humble/setup.bash
colcon build --symlink-install

# Add workspace to bashrc
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

### 2. Clone Essential Repositories
```bash
cd ~/ros2_ws/src

# Clone navigation stack
git clone -b humble https://github.com/ros-planning/navigation2.git

# Clone robot localization
git clone -b humble https://github.com/cra-ros-pkg/robot_localization.git

# Clone vision packages
git clone -b humble https://github.com/ros-perception/vision_opencv.git

# Clone hardware interfaces
git clone -b humble https://github.com/ros-controls/ros2_control.git
git clone -b humble https://github.com/ros-controls/ros2_controllers.git
```

## Testing Your Setup

### 1. Test ROS2 Installation
```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Test ROS2
ros2 run demo_nodes_cpp talker
```

In another terminal:
```bash
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp listener
```

### 2. Test Python Environment
```bash
# Activate virtual environment
source ~/robotics_env/bin/activate

# Test Python libraries
python3 -c "import numpy as np; print('NumPy version:', np.__version__)"
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### 3. Test GPU Acceleration
```bash
# Check GPU
nvidia-smi

# Test PyTorch GPU
source ~/robotics_env/bin/activate
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

## Troubleshooting Common Issues

### 1. ROS2 Installation Issues
```bash
# If ROS2 packages are not found
sudo apt update
sudo apt install python3-rosdep
sudo rosdep init
rosdep update
```

### 2. Python Virtual Environment Issues
```bash
# Recreate virtual environment if needed
deactivate  # if currently in an environment
rm -rf ~/robotics_env
python3 -m venv ~/robotics_env
source ~/robotics_env/bin/activate
```

### 3. Permission Issues with Hardware
```bash
# Add user to dialout group for serial communication
sudo usermod -a -G dialout $USER

# Add user to video group for camera access
sudo usermod -a -G video $USER

# Log out and log back in for changes to take effect
```

## Performance Optimization

### 1. Configure Swap Space
```bash
# Check current swap
free -h

# Create swap file (adjust size as needed)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make swap permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 2. Optimize System for Real-time Performance
```bash
# Install real-time kernel (optional)
sudo apt install -y linux-image-rt-generic

# Configure CPU governor for performance
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## Security Considerations

### 1. Firewall Configuration
```bash
# Enable UFW firewall
sudo ufw enable

# Allow ROS2 communication
sudo ufw allow 11311  # ROS2 master port
sudo ufw allow 5000:5500/udp  # Gazebo ports
```

### 2. Secure Development Practices
- Use virtual environments for Python development
- Keep system and packages updated
- Use secure communication protocols when applicable
- Regularly backup important code and data

## Next Steps

1. **Explore ROS2 tutorials**: Follow the official ROS2 tutorials to get familiar with the framework
2. **Set up simulation environments**: Start with Gazebo or PyBullet to test your development environment
3. **Create your first ROS2 package**: Practice creating and building ROS2 packages
4. **Integrate with hardware**: Once comfortable with simulation, begin integrating with physical hardware

## Additional Resources

- [ROS2 Documentation](https://docs.ros.org/en/humble/)
- [Ubuntu Installation Guide](https://ubuntu.com/tutorials/install-ubuntu-desktop)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [Python Robotics Libraries](https://pypi.org/)
- [Gazebo Simulation](https://gazebosim.org/)