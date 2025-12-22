---
sidebar_position: 3
title: "Chapter 4.3: Multi-Modal Systems"
description: "Understanding and implementing multi-modal systems for robotics with sensor fusion and cross-modal reasoning"
---

# Chapter 4.3: Multi-Modal Systems

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the principles of multi-modal integration in robotics
- Implement sensor fusion techniques for robust perception
- Design cross-modal reasoning systems
- Apply multi-modal learning approaches to robotics tasks
- Evaluate multi-modal system performance in real-world scenarios
- Integrate multiple sensory modalities for enhanced robot capabilities

## Introduction to Multi-Modal Systems

Multi-modal systems in robotics refer to the integration of multiple sensory modalities to enhance perception, reasoning, and action capabilities. These systems combine information from various sensors such as cameras, LiDAR, IMU, tactile sensors, microphones, and more to create a comprehensive understanding of the environment.

### Why Multi-Modal Systems?

Single-sensor systems often face limitations:
- **Visual sensors** can be affected by lighting conditions, occlusions, or reflections
- **LiDAR** may miss transparent or highly reflective objects
- **Tactile sensors** provide local information but lack spatial context
- **Audio sensors** can detect events but lack spatial precision

Multi-modal systems overcome these limitations by combining complementary information sources, leading to:
- Robust perception under various conditions
- Redundant sensing for safety-critical applications
- Enhanced situational awareness
- Improved task performance

### Types of Multi-Modal Integration

1. **Early Fusion**: Raw sensor data is combined at the lowest level
2. **Late Fusion**: Individual sensor outputs are combined at decision level
3. **Deep Fusion**: Integration occurs at multiple levels of abstraction
4. **Cross-Modal Learning**: Information from one modality guides another

## Sensor Fusion Fundamentals

### Kalman Filters for Multi-Sensor Fusion

```python
import numpy as np
from scipy.linalg import inv

class MultiModalKalmanFilter:
    def __init__(self, state_dim, control_dim=0):
        """
        Initialize Kalman filter for multi-sensor fusion
        """
        self.state_dim = state_dim
        self.control_dim = control_dim

        # State vector: [x, y, z, vx, vy, vz, ax, ay, az]
        self.x = np.zeros(state_dim)  # State estimate
        self.P = np.eye(state_dim) * 1000  # State covariance
        self.Q = np.eye(state_dim) * 0.1  # Process noise
        self.R = {}  # Measurement noise for each sensor

        # Time step
        self.dt = 0.01  # 100Hz

    def predict(self, u=None):
        """
        Prediction step: predict next state based on motion model
        """
        # State transition matrix (constant velocity model)
        F = np.eye(self.state_dim)
        F[0, 3] = self.dt  # x = x + vx*dt
        F[1, 4] = self.dt  # y = y + vy*dt
        F[2, 5] = self.dt  # z = z + vz*dt
        F[3, 6] = self.dt  # vx = vx + ax*dt
        F[4, 7] = self.dt  # vy = vy + ay*dt
        F[5, 8] = self.dt  # vz = vz + az*dt

        # Control input matrix (if provided)
        B = np.zeros((self.state_dim, self.control_dim)) if u is not None else None

        # Predict state
        self.x = F @ self.x
        if u is not None:
            self.x += B @ u

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z, H, sensor_type='default'):
        """
        Update step: incorporate measurement from sensor
        """
        if sensor_type not in self.R:
            self.R[sensor_type] = np.eye(len(z)) * 1.0  # Default measurement noise

        # Innovation
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R[sensor_type]
        K = self.P @ H.T @ inv(S)

        # Update state and covariance
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

        return self.x.copy()

class MultiSensorFusionSystem:
    def __init__(self):
        # Initialize Kalman filter for 9D state [pos, vel, acc]
        self.kf = MultiModalKalmanFilter(state_dim=9)

        # Sensor-specific measurement matrices
        self.H_vision = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # x from vision
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # y from vision
            [0, 0, 1, 0, 0, 0, 0, 0, 0]   # z from vision
        ])

        self.H_imu = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0, 0],  # vx from IMU
            [0, 0, 0, 0, 1, 0, 0, 0, 0],  # vy from IMU
            [0, 0, 0, 0, 0, 1, 0, 0, 0],  # vz from IMU
            [0, 0, 0, 0, 0, 0, 1, 0, 0],  # ax from IMU
            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # ay from IMU
            [0, 0, 0, 0, 0, 0, 0, 0, 1]   # az from IMU
        ])

        self.H_lidar = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # x from LiDAR
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # y from LiDAR
            [0, 0, 1, 0, 0, 0, 0, 0, 0]   # z from LiDAR
        ])

    def process_vision_measurement(self, vision_pos):
        """
        Process vision-based position measurement
        """
        self.kf.update(vision_pos, self.H_vision, 'vision')

    def process_imu_measurement(self, imu_vel_acc):
        """
        Process IMU-based velocity and acceleration measurement
        """
        self.kf.update(imu_vel_acc, self.H_imu, 'imu')

    def process_lidar_measurement(self, lidar_pos):
        """
        Process LiDAR-based position measurement
        """
        self.kf.update(lidar_pos, self.H_lidar, 'lidar')

    def get_fused_state(self):
        """
        Get the current fused state estimate
        """
        return self.kf.x

# Example usage
fusion_system = MultiSensorFusionSystem()

# Simulate measurements from different sensors
vision_measurement = np.array([1.0, 2.0, 0.5])  # x, y, z from vision
imu_measurement = np.array([0.1, 0.05, 0.0, 0.0, 0.0, 9.81])  # vx, vy, vz, ax, ay, az from IMU
lidar_measurement = np.array([1.05, 2.02, 0.48])  # x, y, z from LiDAR

# Process measurements
fusion_system.process_vision_measurement(vision_measurement)
fusion_system.process_imu_measurement(imu_measurement)
fusion_system.process_lidar_measurement(lidar_measurement)

# Get fused state
fused_state = fusion_system.get_fused_state()
print(f"Fused state: {fused_state}")
```

### Particle Filters for Non-Linear Multi-Modal Fusion

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class MultiModalParticleFilter:
    def __init__(self, num_particles=1000, state_dim=3):
        """
        Initialize particle filter for multi-modal fusion
        """
        self.num_particles = num_particles
        self.state_dim = state_dim

        # Initialize particles randomly
        self.particles = np.random.uniform(-10, 10, (num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, motion_model):
        """
        Predict particle states using motion model
        """
        for i in range(self.num_particles):
            self.particles[i] = motion_model(self.particles[i])

        # Add process noise
        noise = np.random.normal(0, 0.1, self.particles.shape)
        self.particles += noise

    def update(self, measurement, sensor_model):
        """
        Update particle weights based on measurement
        """
        for i in range(self.num_particles):
            predicted_measurement = sensor_model(self.particles[i])
            # Calculate likelihood
            likelihood = norm.pdf(measurement, predicted_measurement, 0.5).prod()
            self.weights[i] *= likelihood

        # Normalize weights
        self.weights += 1e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)

    def resample(self):
        """
        Resample particles based on weights
        """
        indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        """
        Get state estimate as weighted average
        """
        return np.average(self.particles, axis=0, weights=self.weights)

class MultiSensorParticleFusion:
    def __init__(self):
        self.pf = MultiModalParticleFilter(num_particles=1000, state_dim=6)  # [x, y, z, vx, vy, vz]

    def motion_model(self, state):
        """
        Simple motion model: constant velocity
        """
        dt = 0.01
        new_state = state.copy()
        new_state[0] += state[3] * dt  # x += vx * dt
        new_state[1] += state[4] * dt  # y += vy * dt
        new_state[2] += state[5] * dt  # z += vz * dt
        return new_state

    def vision_sensor_model(self, state):
        """
        Vision sensor model (returns expected 2D position)
        """
        # For simplicity, return x, y coordinates
        return state[:2]

    def lidar_sensor_model(self, state):
        """
        LiDAR sensor model (returns expected 3D position)
        """
        return state[:3]

    def process_measurements(self, vision_meas, lidar_meas):
        """
        Process measurements from multiple sensors
        """
        # Predict step
        self.pf.predict(self.motion_model)

        # Update with vision measurement
        self.pf.update(vision_meas, self.vision_sensor_model)

        # Update with LiDAR measurement
        self.pf.update(lidar_meas, self.lidar_sensor_model)

        # Resample if effective sample size is low
        if 1.0 / np.sum(self.pf.weights**2) < self.pf.num_particles / 2:
            self.pf.resample()

        return self.pf.estimate()

# Example usage
fusion = MultiSensorParticleFusion()

# Simulate measurements
for t in range(100):
    vision_meas = np.array([t*0.1 + np.random.normal(0, 0.05),
                           t*0.05 + np.random.normal(0, 0.05)])
    lidar_meas = np.array([t*0.1 + np.random.normal(0, 0.02),
                          t*0.05 + np.random.normal(0, 0.02),
                          1.0 + np.random.normal(0, 0.01)])

    estimate = fusion.process_measurements(vision_meas, lidar_meas)
    print(f"Step {t}: Estimated position = {estimate[:3]}")
```

## Deep Learning Approaches to Multi-Modal Fusion

### Cross-Modal Attention Networks

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim=512):
        super(CrossModalAttention, self).__init__()

        # Linear projections
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Attention mechanisms
        self.visual_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        self.text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )

        # Cross-attention
        self.cross_attn_vt = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        self.cross_attn_tv = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, visual_features, text_features):
        # Project features
        v_proj = self.visual_proj(visual_features)  # (seq_len_v, batch, hidden_dim)
        t_proj = self.text_proj(text_features)      # (seq_len_t, batch, hidden_dim)

        # Self-attention within modalities
        v_self, _ = self.visual_attn(v_proj, v_proj, v_proj)
        t_self, _ = self.text_attn(t_proj, t_proj, t_proj)

        # Cross-attention: visual attends to text
        v_cross, _ = self.cross_attn_vt(t_proj, v_proj, v_proj)

        # Cross-attention: text attends to visual
        t_cross, _ = self.cross_attn_tv(v_proj, t_proj, t_proj)

        # Combine attended features
        combined_v = torch.cat([v_self, v_cross], dim=-1)
        combined_t = torch.cat([t_self, t_cross], dim=-1)

        # Project to output dimension
        output_v = self.output_proj(combined_v)
        output_t = self.output_proj(combined_t)

        return output_v, output_t

class MultiModalFusionNetwork(nn.Module):
    def __init__(self, visual_dim=2048, text_dim=768, tactile_dim=128, output_dim=512):
        super(MultiModalFusionNetwork, self).__init__()

        # Cross-modal attention for vision-text
        self.vt_attention = CrossModalAttention(visual_dim, text_dim)

        # Separate processing for tactile
        self.tactile_processor = nn.Sequential(
            nn.Linear(tactile_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )

        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * 3, 1024),  # Combined features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        # Attention over modalities
        self.modality_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1
        )

    def forward(self, visual_features, text_features, tactile_features):
        # Process vision-text cross-attention
        fused_v, fused_t = self.vt_attention(visual_features, text_features)

        # Process tactile features
        tactile_processed = self.tactile_processor(tactile_features)

        # Take mean of sequence dimensions for each modality
        if len(fused_v.shape) > 2:
            fused_v = fused_v.mean(dim=0)  # Average over sequence
        if len(fused_t.shape) > 2:
            fused_t = fused_t.mean(dim=0)  # Average over sequence

        # Concatenate all modalities
        combined = torch.cat([fused_v, fused_t, tactile_processed], dim=-1)

        # Final fusion
        fused_output = self.fusion_layer(combined)

        return fused_output

# Example usage
def create_sample_data():
    """
    Create sample multi-modal data
    """
    batch_size = 4
    seq_len_v = 10  # Visual sequence length
    seq_len_t = 20  # Text sequence length

    visual_features = torch.randn(seq_len_v, batch_size, 2048)  # Visual features
    text_features = torch.randn(seq_len_t, batch_size, 768)     # Text features
    tactile_features = torch.randn(batch_size, 128)             # Tactile features

    return visual_features, text_features, tactile_features

# Initialize model and test
model = MultiModalFusionNetwork()
visual, text, tactile = create_sample_data()

output = model(visual, text, tactile)
print(f"Fused output shape: {output.shape}")
```

### Late Fusion with Attention

```python
class LateFusionWithAttention(nn.Module):
    def __init__(self, num_modalities, feature_dim, output_dim):
        super(LateFusionWithAttention, self).__init__()

        self.num_modalities = num_modalities
        self.feature_dim = feature_dim

        # Individual modality encoders
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, output_dim),
                nn.ReLU()
            ) for _ in range(num_modalities)
        ])

        # Attention over modalities
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1
        )

        # Final classifier
        self.classifier = nn.Linear(output_dim, output_dim)

    def forward(self, modalities):
        """
        modalities: List of features from different modalities
        """
        encoded_features = []

        # Encode each modality
        for i, modality in enumerate(modalities):
            encoded = self.encoders[i](modality)
            encoded_features.append(encoded)

        # Stack encoded features
        stacked_features = torch.stack(encoded_features, dim=0)  # (num_modalities, batch, output_dim)

        # Apply attention over modalities
        attended_features, attention_weights = self.attention(
            stacked_features, stacked_features, stacked_features
        )

        # Take mean across modalities
        fused_features = attended_features.mean(dim=0)

        # Final classification
        output = self.classifier(fused_features)

        return output, attention_weights

# Example usage
late_fusion = LateFusionWithAttention(
    num_modalities=3,  # Vision, text, tactile
    feature_dim=512,
    output_dim=256
)

# Sample modalities
vision_features = torch.randn(32, 512)
text_features = torch.randn(32, 512)
tactile_features = torch.randn(32, 512)

modalities = [vision_features, text_features, tactile_features]
output, attention_weights = late_fusion(modalities)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
```

## Multi-Modal Learning Frameworks

### Vision-Text-Tactile Integration

```python
class VisionTextTactileFusion(nn.Module):
    def __init__(self, vision_dim=2048, text_dim=768, tactile_dim=128, output_dim=512):
        super(VisionTextTactileFusion, self).__init__()

        # Vision encoder
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        # Tactile encoder
        self.tactile_encoder = nn.Sequential(
            nn.Linear(tactile_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU()
        )

        # Cross-modal attention layers
        self.vt_cross_attention = CrossModalAttention(512, 512)
        self.vt_tactile_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.1
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(512 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, vision_features, text_features, tactile_features):
        # Encode each modality
        vision_encoded = self.vision_encoder(vision_features)
        text_encoded = self.text_encoder(text_features)
        tactile_encoded = self.tactile_encoder(tactile_features)

        # Reshape for attention (required format: seq_len, batch, embed_dim)
        vision_encoded = vision_encoded.unsqueeze(0)  # (1, batch, 512)
        text_encoded = text_encoded.unsqueeze(0)      # (1, batch, 512)
        tactile_encoded = tactile_encoded.unsqueeze(0)  # (1, batch, 512)

        # Cross-attention between vision and text
        vision_text_v, vision_text_t = self.vt_cross_attention(
            vision_encoded, text_encoded
        )

        # Attention with tactile modality
        all_modalities = torch.cat([
            vision_text_v,
            vision_text_t,
            tactile_encoded
        ], dim=0)  # (3, batch, 512)

        # Final attention fusion
        attended, _ = self.vt_tactile_attention(
            all_modalities, all_modalities, all_modalities
        )

        # Reshape and concatenate
        attended_flat = attended.transpose(0, 1).reshape(-1, 512 * 3)  # (batch, 512*3)

        # Final fusion
        output = self.fusion(attended_flat)

        return output

# Example usage in a robotics context
class RobotMultiModalPerception:
    def __init__(self):
        self.fusion_model = VisionTextTactileFusion()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fusion_model.to(self.device)

    def process_perception(self, visual_input, text_instruction, tactile_input):
        """
        Process multi-modal inputs for robotic perception
        """
        # Move inputs to device
        visual_input = visual_input.to(self.device)
        text_instruction = text_instruction.to(self.device)
        tactile_input = tactile_input.to(self.device)

        # Forward pass
        with torch.no_grad():
            fused_features = self.fusion_model(visual_input, text_instruction, tactile_input)

        return fused_features

    def execute_task(self, visual_obs, text_cmd, tactile_sense):
        """
        Execute robotic task based on multi-modal perception
        """
        # Process multi-modal inputs
        fused_state = self.process_perception(visual_obs, text_cmd, tactile_sense)

        # Generate action based on fused state
        action = self.generate_action(fused_state)

        return action

    def generate_action(self, fused_state):
        """
        Generate robot action from fused state representation
        """
        # This would typically connect to robot control system
        # For now, return a dummy action
        action_dim = 7  # 7-DoF robot arm
        return torch.randn(action_dim).to(fused_state.device)

# Example usage
robot_perception = RobotMultiModalPerception()

# Sample inputs
batch_size = 8
visual_input = torch.randn(batch_size, 2048)    # Visual features
text_input = torch.randn(batch_size, 768)       # Text features
tactile_input = torch.randn(batch_size, 128)    # Tactile features

# Process multi-modal perception
fused_output = robot_perception.process_perception(
    visual_input, text_input, tactile_input
)

print(f"Multi-modal fused output shape: {fused_output.shape}")
```

## Implementation with ROS2

### Multi-Modal Sensor Integration Node

```python
#!/usr/bin/env python3
"""
ROS2 node for multi-modal sensor fusion
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Imu, JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import numpy as np
import torch
from torchvision import transforms
from cv_bridge import CvBridge
import message_filters
from sensor_msgs_py import point_cloud2

class MultiModalFusionNode(Node):
    def __init__(self):
        super().__init__('multi_modal_fusion_node')

        # Initialize CvBridge for image processing
        self.bridge = CvBridge()

        # Create subscribers for different sensors
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, '/lidar/points')
        self.imu_sub = message_filters.Subscriber(self, Imu, '/imu/data')
        self.joint_sub = message_filters.Subscriber(self, JointState, '/joint_states')

        # Synchronize messages from different sensors
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub, self.imu_sub, self.joint_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.multi_modal_callback)

        # Publisher for fused state
        self.fused_pub = self.create_publisher(PoseStamped, '/fused_state', 10)

        # Publisher for commands
        self.cmd_pub = self.create_publisher(String, '/robot_command', 10)

        # Initialize fusion model
        self.fusion_model = VisionTextTactileFusion()
        self.fusion_model.eval()

        self.get_logger().info('Multi-modal fusion node initialized')

    def multi_modal_callback(self, image_msg, lidar_msg, imu_msg, joint_msg):
        """
        Callback for synchronized multi-modal sensor data
        """
        try:
            # Process image data
            image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            visual_features = self.extract_visual_features(image)

            # Process LiDAR data
            lidar_points = list(point_cloud2.read_points(
                lidar_msg,
                field_names=('x', 'y', 'z'),
                skip_nans=True
            ))
            lidar_features = self.extract_lidar_features(lidar_points)

            # Process IMU data
            imu_features = self.extract_imu_features(imu_msg)

            # Process joint states
            joint_features = self.extract_joint_features(joint_msg)

            # Combine all features for fusion
            fused_state = self.perform_fusion(
                visual_features,
                lidar_features,
                imu_features,
                joint_features
            )

            # Publish fused state
            self.publish_fused_state(fused_state)

        except Exception as e:
            self.get_logger().error(f'Error in multi-modal callback: {str(e)}')

    def extract_visual_features(self, image):
        """
        Extract visual features using pre-trained model
        """
        # Convert image to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Use a pre-trained CNN to extract features
        import torchvision.models as models
        feature_extractor = models.resnet50(pretrained=True)
        feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
        feature_extractor.eval()

        with torch.no_grad():
            features = feature_extractor(image_tensor)
            features = features.view(features.size(0), -1)  # Flatten

        return features.squeeze(0).cpu().numpy()  # Remove batch dim and convert to numpy

    def extract_lidar_features(self, points):
        """
        Extract features from LiDAR point cloud
        """
        if len(points) == 0:
            return np.zeros(128)  # Return zero features if no points

        points_array = np.array(points)

        # Basic statistical features
        features = []
        features.extend([
            np.mean(points_array[:, 0]),  # Mean X
            np.mean(points_array[:, 1]),  # Mean Y
            np.mean(points_array[:, 2]),  # Mean Z
            np.std(points_array[:, 0]),   # Std X
            np.std(points_array[:, 1]),   # Std Y
            np.std(points_array[:, 2]),   # Std Z
            len(points_array),            # Number of points
            np.min(points_array[:, 2]),   # Min Z (height)
            np.max(points_array[:, 2]),   # Max Z (height)
        ])

        # Pad or truncate to fixed size
        features = np.array(features)
        if len(features) < 128:
            features = np.pad(features, (0, 128 - len(features)), mode='constant')
        else:
            features = features[:128]

        return features

    def extract_imu_features(self, imu_msg):
        """
        Extract features from IMU data
        """
        features = np.array([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z,
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z,
        ])

        return features

    def extract_joint_features(self, joint_msg):
        """
        Extract features from joint states
        """
        if len(joint_msg.position) >= 7:
            # Use first 7 joints (for 7-DoF arm)
            features = np.array(joint_msg.position[:7])
        else:
            # Pad with zeros if fewer joints
            features = np.array(joint_msg.position + [0.0] * (7 - len(joint_msg.position)))

        return features

    def perform_fusion(self, visual_features, lidar_features, imu_features, joint_features):
        """
        Perform multi-modal fusion
        """
        # For simplicity, we'll use a weighted average approach
        # In practice, you'd use a learned fusion model

        # Normalize features to same scale
        visual_norm = visual_features / (np.linalg.norm(visual_features) + 1e-8)
        lidar_norm = lidar_features / (np.linalg.norm(lidar_features) + 1e-8)
        imu_norm = imu_features / (np.linalg.norm(imu_features) + 1e-8)
        joint_norm = joint_features / (np.linalg.norm(joint_features) + 1e-8)

        # Weighted fusion (weights can be learned)
        weights = [0.4, 0.3, 0.15, 0.15]  # Visual, LiDAR, IMU, Joint
        fused_state = (
            weights[0] * visual_norm +
            weights[1] * lidar_norm +
            weights[2] * imu_norm +
            weights[3] * joint_norm
        )

        return fused_state

    def publish_fused_state(self, fused_state):
        """
        Publish the fused state as a PoseStamped message
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'fused_state'

        # Use first 3 elements as position (x, y, z)
        pose_msg.pose.position.x = float(fused_state[0]) if len(fused_state) > 0 else 0.0
        pose_msg.pose.position.y = float(fused_state[1]) if len(fused_state) > 1 else 0.0
        pose_msg.pose.position.z = float(fused_state[2]) if len(fused_state) > 2 else 0.0

        # Use next 4 elements as orientation (quaternion)
        if len(fused_state) >= 6:
            pose_msg.pose.orientation.x = float(fused_state[3])
            pose_msg.pose.orientation.y = float(fused_state[4])
            pose_msg.pose.orientation.z = float(fused_state[5])
            pose_msg.pose.orientation.w = float(fused_state[6]) if len(fused_state) > 6 else 1.0
        else:
            pose_msg.pose.orientation.w = 1.0  # Default orientation

        self.fused_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)

    fusion_node = MultiModalFusionNode()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Multi-Modal Training Strategies

### Contrastive Learning for Multi-Modal Alignment

```python
class MultiModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(MultiModalContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, vision_features, text_features):
        """
        Compute contrastive loss between vision and text features
        """
        # Normalize features
        vision_features = F.normalize(vision_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(vision_features, text_features.T) / self.temperature

        # Create labels (diagonal elements are positive pairs)
        batch_size = vision_features.shape[0]
        labels = torch.arange(batch_size).to(vision_features.device)

        # Compute cross-entropy loss
        loss_vision = F.cross_entropy(similarity_matrix, labels)
        loss_text = F.cross_entropy(similarity_matrix.T, labels)

        # Return average loss
        return (loss_vision + loss_text) / 2

class MultiModalContrastiveModel(nn.Module):
    def __init__(self, vision_dim=2048, text_dim=768, projection_dim=512):
        super(MultiModalContrastiveModel, self).__init__()

        # Vision encoder
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, projection_dim)
        )

        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, projection_dim)
        )

        # Projection heads
        self.vision_projection = nn.Linear(projection_dim, projection_dim)
        self.text_projection = nn.Linear(projection_dim, projection_dim)

    def encode_vision(self, vision_features):
        vision_embeds = self.vision_encoder(vision_features)
        vision_embeds = F.normalize(vision_embeds, dim=-1)
        vision_embeds = self.vision_projection(vision_embeds)
        return vision_embeds

    def encode_text(self, text_features):
        text_embeds = self.text_encoder(text_features)
        text_embeds = F.normalize(text_embeds, dim=-1)
        text_embeds = self.text_projection(text_embeds)
        return text_embeds

    def forward(self, vision_features, text_features):
        vision_embeds = self.encode_vision(vision_features)
        text_embeds = self.encode_text(text_features)

        return vision_embeds, text_embeds

# Training loop example
def train_multimodal_contrastive(model, dataloader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = MultiModalContrastiveLoss(temperature=0.07)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (vision_batch, text_batch) in enumerate(dataloader):
            vision_batch = vision_batch.to(device)
            text_batch = text_batch.to(device)

            # Forward pass
            vision_embeds, text_embeds = model(vision_batch, text_batch)

            # Compute loss
            loss = criterion(vision_embeds, text_embeds)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
```

## Evaluation Metrics for Multi-Modal Systems

```python
class MultiModalEvaluator:
    def __init__(self):
        self.metrics = {}

    def compute_modality_alignment(self, vision_features, text_features):
        """
        Compute alignment between vision and text modalities
        """
        # Compute cosine similarity between vision and text features
        vision_norm = F.normalize(vision_features, dim=1)
        text_norm = F.normalize(text_features, dim=1)

        similarity_matrix = torch.matmul(vision_norm, text_norm.T)

        # Compute diagonal accuracy (how well aligned are corresponding pairs)
        diagonal_matches = torch.diag(similarity_matrix)
        non_diagonal = similarity_matrix - torch.diag_embed(diagonal_matches)

        # Alignment score: how much higher are diagonal elements than off-diagonal
        alignment_score = torch.mean(diagonal_matches - torch.mean(non_diagonal, dim=1))

        return alignment_score.item()

    def compute_fusion_effectiveness(self, individual_scores, fused_score):
        """
        Compute how much fusion improves over individual modalities
        """
        baseline_score = max(individual_scores)  # Best single modality
        improvement = fused_score - baseline_score

        effectiveness = improvement / (baseline_score + 1e-8)  # Avoid division by zero
        return effectiveness

    def compute_redundancy(self, modality_features):
        """
        Compute redundancy between modalities
        """
        correlations = []
        for i in range(len(modality_features)):
            for j in range(i+1, len(modality_features)):
                # Compute correlation between modalities
                feat1 = modality_features[i].flatten()
                feat2 = modality_features[j].flatten()

                # Pearson correlation
                mean1, mean2 = torch.mean(feat1), torch.mean(feat2)
                std1, std2 = torch.std(feat1), torch.std(feat2)

                correlation = torch.mean((feat1 - mean1) * (feat2 - mean2)) / (std1 * std2 + 1e-8)
                correlations.append(torch.abs(correlation))

        avg_redundancy = torch.mean(torch.tensor(correlations)) if correlations else 0.0
        return avg_redundancy.item()

# Example evaluation
evaluator = MultiModalEvaluator()

# Sample features from different modalities
vision_features = torch.randn(32, 512)
text_features = torch.randn(32, 512)

alignment = evaluator.compute_modality_alignment(vision_features, text_features)
print(f"Modality alignment score: {alignment:.4f}")
```

## Hands-on Exercises

### Exercise 1: Implement Sensor Fusion for Robot Navigation
1. Implement a Kalman filter that fuses data from wheel encoders, IMU, and visual odometry
2. Test the system in simulation with noisy sensor data
3. Compare the fused estimate with individual sensor estimates
4. Evaluate the improvement in navigation accuracy

### Exercise 2: Multi-Modal Object Recognition
1. Create a dataset with images and corresponding tactile sensor readings
2. Implement a multi-modal network that combines visual and tactile features
3. Train the network to recognize objects using both modalities
4. Evaluate performance compared to single-modality approaches

### Exercise 3: Cross-Modal Attention for Robotics
1. Implement a cross-modal attention mechanism for vision-language tasks
2. Train on a dataset of images with natural language descriptions
3. Test the system on robotic manipulation tasks
4. Analyze attention patterns to understand cross-modal relationships

### Exercise 4: Real-time Multi-Modal Fusion
1. Set up real-time data streams from multiple sensors
2. Implement efficient fusion algorithms suitable for real-time operation
3. Deploy on a robotic platform
4. Evaluate computational efficiency and accuracy trade-offs

## Best Practices for Multi-Modal Systems

### Data Synchronization
- Use hardware or software timestamps to align sensor data
- Implement buffer mechanisms to handle different sensor rates
- Apply interpolation for temporal alignment
- Consider causality and latency requirements

### Model Architecture Selection
- Choose fusion strategy based on task requirements
- Consider computational constraints for deployment
- Evaluate trade-offs between accuracy and efficiency
- Plan for scalability to additional modalities

### Robustness Considerations
- Handle missing or corrupted sensor data gracefully
- Implement fallback mechanisms for sensor failures
- Design for various environmental conditions
- Test extensively under diverse operating conditions

## Review Questions

1. What are the main advantages of multi-modal systems over single-sensor approaches?
2. Explain the differences between early, late, and deep fusion strategies.
3. How does cross-modal attention work in multi-modal neural networks?
4. What are the key challenges in synchronizing data from multiple sensors?
5. How can you evaluate the effectiveness of multi-modal fusion?
6. What are the computational trade-offs in multi-modal system design?
7. Describe how Kalman filters can be used for sensor fusion.
8. What are the key considerations for deploying multi-modal systems on robots?

## Further Reading and Resources

- "Deep Multimodal Learning: A Survey" - Comprehensive survey of multi-modal learning approaches
- "Vision-Language Models for Vision Tasks: A Survey" - Latest developments in VLMs
- "Sensor Fusion for Robotics: A Survey" - Traditional and modern sensor fusion techniques
- "Cross-Modal Learning in Robotics" - Applications of cross-modal learning to robotics
- "Multimodal Deep Learning" - Foundational concepts in multi-modal deep learning
- ROS2 documentation on multi-sensor integration and message synchronization
- "Probabilistic Robotics" by Thrun, Burgard, and Fox - Classical approaches to sensor fusion