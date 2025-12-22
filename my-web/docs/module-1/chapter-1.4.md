---
sidebar_position: 4
title: "Chapter 1.4: URDF for Humanoids"
description: "Understanding Unified Robot Description Format for modeling humanoid robots in ROS2"
---

# Chapter 1.4: URDF for Humanoids

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the Unified Robot Description Format (URDF) structure
- Create robot models for humanoid robots
- Define joints, links, and kinematic chains
- Simulate humanoid robots in Gazebo and RViz
- Apply URDF best practices for humanoid robotics

## Introduction to URDF

Unified Robot Description Format (URDF) is an XML-based format used in ROS to describe robot models. It defines the physical and visual properties of a robot, including its links, joints, and how they connect. URDF is essential for simulation, visualization, and kinematic analysis of robots.

URDF files contain:
- **Links**: Rigid bodies with physical properties (mass, inertia, visual/collision geometry)
- **Joints**: Connections between links that define degrees of freedom
- **Materials**: Visual appearance of links
- **Transmissions**: Mapping between joints and actuators
- **Gazebo plugins**: Simulation-specific configurations

For humanoid robots, URDF becomes more complex due to the multiple limbs and intricate kinematic structures.

## URDF Structure for Humanoid Robots

Humanoid robots typically follow a standard structure:
- **Torso**: Main body containing the center of mass
- **Head**: Contains sensors (cameras, IMU, etc.)
- **Arms**: Left and right arms with multiple joints
- **Legs**: Left and right legs for locomotion
- **Hands**: End effectors for manipulation

### Basic URDF Template

Here's a minimal URDF template for a humanoid robot:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Torso Link -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <!-- Head Link -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Joint connecting torso and head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.35" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Example: Right Arm Links and Joints -->
  <link name="upper_arm_right">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="shoulder_right_joint" type="revolute">
    <parent link="torso"/>
    <child link="upper_arm_right"/>
    <origin xyz="0.1 -0.15 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="1.5" effort="50" velocity="2"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Example: Right Lower Arm -->
  <link name="lower_arm_right">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.008" ixy="0" ixz="0" iyy="0.008" iyz="0" izz="0.004"/>
    </inertial>
  </link>

  <joint name="elbow_right_joint" type="revolute">
    <parent link="upper_arm_right"/>
    <child link="lower_arm_right"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.5" effort="30" velocity="2"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Example: Left Arm Links and Joints -->
  <link name="upper_arm_left">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="shoulder_left_joint" type="revolute">
    <parent link="torso"/>
    <child link="upper_arm_left"/>
    <origin xyz="0.1 0.15 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="2.0" effort="50" velocity="2"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="lower_arm_left">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.008" ixy="0" ixz="0" iyy="0.008" iyz="0" izz="0.004"/>
    </inertial>
  </link>

  <joint name="elbow_left_joint" type="revolute">
    <parent link="upper_arm_left"/>
    <child link="lower_arm_left"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.5" effort="30" velocity="2"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Example: Right Leg Links and Joints -->
  <link name="hip_right">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.15"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="hip_right_joint" type="revolute">
    <parent link="torso"/>
    <child link="hip_right"/>
    <origin xyz="-0.05 -0.1 -0.25" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.78" upper="0.78" effort="100" velocity="1"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="thigh_right">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3"/>
      <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="knee_right_joint" type="revolute">
    <parent link="hip_right"/>
    <child link="thigh_right"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.0" effort="150" velocity="1"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="shin_right">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.5"/>
      <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="ankle_right_joint" type="revolute">
    <parent link="thigh_right"/>
    <child link="shin_right"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" effort="100" velocity="1"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="foot_right">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.003"/>
    </inertial>
  </link>

  <joint name="foot_right_joint" type="fixed">
    <parent link="shin_right"/>
    <child link="foot_right"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
  </joint>

  <!-- Example: Left Leg Links and Joints -->
  <link name="hip_left">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.15"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="hip_left_joint" type="revolute">
    <parent link="torso"/>
    <child link="hip_left"/>
    <origin xyz="-0.05 0.1 -0.25" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.78" upper="0.78" effort="100" velocity="1"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="thigh_left">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3"/>
      <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="knee_left_joint" type="revolute">
    <parent link="hip_left"/>
    <child link="thigh_left"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.0" effort="150" velocity="1"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="shin_left">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.5"/>
      <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="ankle_left_joint" type="revolute">
    <parent link="thigh_left"/>
    <child link="shin_left"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" effort="100" velocity="1"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="foot_left">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.003"/>
    </inertial>
  </link>

  <joint name="foot_left_joint" type="fixed">
    <parent link="shin_left"/>
    <child link="foot_left"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
  </joint>
</robot>
```

## URDF Joints and Their Types

URDF supports several joint types, each suitable for different robotic applications:

### Joint Types:
- **Revolute**: Rotational joint with limited range
- **Continuous**: Rotational joint without limits
- **Prismatic**: Linear sliding joint with limits
- **Fixed**: No movement, rigid connection
- **Floating**: 6-DOF with no constraints
- **Planar**: Movement constrained to a plane

For humanoid robots, revolute joints are most common, allowing controlled rotation at joints like elbows, knees, and shoulders.

## Advanced URDF Concepts

### Xacro for Complex Robots

Xacro (XML Macros) extends URDF by allowing macros, variables, and mathematical expressions. This is essential for humanoid robots with repetitive structures:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <!-- Define a macro for a generic arm link -->
  <xacro:macro name="arm_link" params="name radius length mass position">
    <link name="${name}">
      <visual>
        <geometry>
          <cylinder length="${length}" radius="${radius}"/>
        </geometry>
        <material name="gray">
          <color rgba="0.5 0.5 0.5 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder length="${length}" radius="${radius}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="${mass}"/>
        <inertia ixx="${mass*(3*radius*radius + length*length)/12}"
                 ixy="0" ixz="0"
                 iyy="${mass*(3*radius*radius + length*length)/12}"
                 iyz="0" izz="${mass*radius*radius/2}"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Define a macro for a revolute joint -->
  <xacro:macro name="revolute_joint" params="name parent child xyz axis lower upper">
    <joint name="${name}" type="revolute">
      <parent link="${parent}"/>
      <child link="${child}"/>
      <origin xyz="${xyz}" rpy="0 0 0"/>
      <axis xyz="${axis}"/>
      <limit lower="${lower}" upper="${upper}" effort="100" velocity="2"/>
      <dynamics damping="0.1" friction="0.0"/>
    </joint>
  </xacro:macro>

  <!-- Use the macros to create an arm -->
  <xacro:arm_link name="upper_arm_right" radius="0.05" length="0.3" mass="1.5" position="right"/>
  <xacro:arm_link name="lower_arm_right" radius="0.04" length="0.25" mass="1.0" position="right"/>

  <xacro:revolute_joint name="shoulder_right_joint"
                        parent="torso"
                        child="upper_arm_right"
                        xyz="0.1 -0.15 -0.1"
                        axis="0 1 0"
                        lower="-2.0"
                        upper="1.5"/>

  <xacro:revolute_joint name="elbow_right_joint"
                        parent="upper_arm_right"
                        child="lower_arm_right"
                        xyz="0 0 -0.3"
                        axis="0 1 0"
                        lower="0"
                        upper="2.5"/>

</robot>
```

### Gazebo Integration

To simulate your humanoid robot in Gazebo, add Gazebo-specific tags to your URDF:

```xml
<gazebo reference="torso">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
</gazebo>

<!-- Controller plugin for Gazebo -->
<gazebo>
  <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
    <robotNamespace>/humanoid</robotNamespace>
    <jointName>shoulder_right_joint, elbow_right_joint, hip_right_joint</jointName>
  </plugin>
</gazebo>
```

## Working with URDF Files

### Loading URDF in RViz
```bash
# Launch RViz with your URDF
ros2 run rviz2 rviz2 -d /path/to/urdf.rviz

# Or create a launch file to visualize the robot
ros2 launch your_package visualize.launch.py
```

### Validating URDF Files
```bash
# Check if URDF is syntactically correct
check_urdf /path/to/your/robot.urdf

# View the kinematic chain
urdf_to_graphviz /path/to/your/robot.urdf
```

## Best Practices for Humanoid URDF

1. **Use proper units**: Always use SI units (meters, kilograms, seconds)

2. **Accurate inertial properties**: Estimate masses and moments of inertia realistically

3. **Consistent naming**: Use clear, descriptive names for links and joints

4. **Realistic joint limits**: Set joint limits based on physical capabilities

5. **Collision vs Visual**: Keep collision geometries simple for performance, make visual geometries detailed

6. **Use Xacro for complex robots**: Reduce redundancy and improve maintainability

7. **Ground truth origins**: Place origins at meaningful locations (joint centers, COM, etc.)

## Python Scripts for URDF Manipulation

Here's a Python script to programmatically create and manipulate URDF:

```python
#!/usr/bin/env python3
"""
URDF Generator for Humanoid Robots

This script demonstrates programmatic creation of URDF for humanoid robots.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import math


class URDFGenerator:
    """
    A class to generate URDF for humanoid robots.
    """

    def __init__(self, robot_name="humanoid_robot"):
        self.robot_name = robot_name
        self.root = ET.Element("robot", name=robot_name)
        self.links = {}
        self.joints = {}

    def add_link(self, name, mass, inertia_matrix, visual_geom, collision_geom,
                 material_rgba=(0.5, 0.5, 0.5, 1)):
        """Add a link to the robot."""
        link = ET.SubElement(self.root, "link", name=name)

        # Visual element
        visual = ET.SubElement(link, "visual")
        vis_geom = ET.SubElement(visual, "geometry")
        vis_geom.append(self._create_geometry_element(visual_geom))

        material = ET.SubElement(visual, "material", name=f"mat_{name}")
        color = ET.SubElement(material, "color",
                             rgba=f"{material_rgba[0]} {material_rgba[1]} {material_rgba[2]} {material_rgba[3]}")

        # Collision element
        collision = ET.SubElement(link, "collision")
        coll_geom = ET.SubElement(collision, "geometry")
        coll_geom.append(self._create_geometry_element(collision_geom))

        # Inertial element
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", value=str(mass))

        inertia_el = ET.SubElement(inertial, "inertia",
                                  ixx=str(inertia_matrix[0][0]),
                                  ixy=str(inertia_matrix[0][1]),
                                  ixz=str(inertia_matrix[0][2]),
                                  iyy=str(inertia_matrix[1][1]),
                                  iyz=str(inertia_matrix[1][2]),
                                  izz=str(inertia_matrix[2][2]))

        self.links[name] = link
        return link

    def _create_geometry_element(self, geom_params):
        """Create geometry element based on parameters."""
        geom_type, params = geom_params
        geom_el = ET.Element(geom_type)

        if geom_type == "box":
            ET.SubElement(geom_el, "size",
                         x=str(params["size"][0]),
                         y=str(params["size"][1]),
                         z=str(params["size"][2]))
        elif geom_type == "cylinder":
            ET.SubElement(geom_el, "radius", value=str(params["radius"]))
            ET.SubElement(geom_el, "length", value=str(params["length"]))
        elif geom_type == "sphere":
            ET.SubElement(geom_el, "radius", value=str(params["radius"]))

        return geom_el

    def add_joint(self, name, joint_type, parent, child, origin_xyz,
                  origin_rpy, axis_xyz=(0, 0, 1), limit_lower=-math.pi,
                  limit_upper=math.pi, effort=100, velocity=1):
        """Add a joint to the robot."""
        joint = ET.SubElement(self.root, "joint", name=name, type=joint_type)

        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)

        origin = ET.SubElement(joint, "origin",
                              xyz=f"{origin_xyz[0]} {origin_xyz[1]} {origin_xyz[2]}",
                              rpy=f"{origin_rpy[0]} {origin_rpy[1]} {origin_rpy[2]}")

        ET.SubElement(joint, "axis", xyz=f"{axis_xyz[0]} {axis_xyz[1]} {axis_xyz[2]}")

        if joint_type in ["revolute", "prismatic"]:
            limit = ET.SubElement(joint, "limit",
                                 lower=str(limit_lower),
                                 upper=str(limit_upper),
                                 effort=str(effort),
                                 velocity=str(velocity))

        dynamics = ET.SubElement(joint, "dynamics", damping="0.1", friction="0.0")

        self.joints[name] = joint
        return joint

    def save_urdf(self, filename):
        """Save the URDF to a file."""
        rough_string = ET.tostring(self.root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        # Remove extra blank lines
        lines = pretty_xml.split('\n')
        non_blank_lines = [line for line in lines if line.strip()]
        pretty_xml = '\n'.join(non_blank_lines)

        with open(filename, 'w') as f:
            f.write(pretty_xml)

        print(f"URDF saved to {filename}")


def main():
    """Main function to demonstrate URDF generation."""
    # Create URDF generator
    urdf_gen = URDFGenerator("programmatic_humanoid")

    # Add torso
    urdf_gen.add_link("torso",
                      mass=10.0,
                      inertia_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                      visual_geom=("box", {"size": [0.3, 0.2, 0.5]}),
                      collision_geom=("box", {"size": [0.3, 0.2, 0.5]}),
                      material_rgba=(0.0, 0.0, 1.0, 1.0))  # Blue

    # Add head
    urdf_gen.add_link("head",
                      mass=2.0,
                      inertia_matrix=[[0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.02]],
                      visual_geom=("sphere", {"radius": 0.1}),
                      collision_geom=("sphere", {"radius": 0.1}),
                      material_rgba=(1.0, 1.0, 1.0, 1.0))  # White

    # Add neck joint
    urdf_gen.add_joint("neck_joint", "revolute", "torso", "head",
                      origin_xyz=[0, 0, 0.35], origin_rpy=[0, 0, 0],
                      axis_xyz=[0, 1, 0],  # Y-axis rotation
                      limit_lower=-math.pi/2, limit_upper=math.pi/2)

    # Save the URDF
    urdf_gen.save_urdf("generated_humanoid.urdf")


if __name__ == "__main__":
    main()
```

## Visualization and Debugging

### Checking URDF in RViz:
```bash
# Visualize the robot model
ros2 run rviz2 rviz2

# In RViz, add RobotModel display and set Robot Description to your robot's description parameter
```

### Validating kinematics:
```bash
# Use kinematics and dynamics library to check the robot
ros2 run kdl_parser check_kdl_urdf your_robot.urdf
```

## Common Issues and Solutions

1. **Joint Limits Too Restrictive**: Set realistic limits based on the physical robot
2. **Inertial Properties Wrong**: Use CAD software to calculate accurate inertias
3. **Links Colliding**: Adjust collision geometries and origins
4. **Kinematic Chain Problems**: Verify parent-child relationships
5. **Simulation Instabilities**: Tune joint damping and stiffness parameters

## Exercises

1. **URDF Creation**:
   - Create a simple 6-DOF arm using the URDF template provided
   - Add proper inertial properties to each link
   - Visualize the arm in RViz

2. **Xacro Exercise**:
   - Convert the basic humanoid URDF to use Xacro macros
   - Create reusable macros for arms and legs
   - Add a parameterized gripper macro

3. **Joint Configuration**:
   - Add realistic joint limits to your humanoid robot
   - Experiment with different joint types (continuous vs revolute)
   - Create a configuration that enables walking

4. **Simulation Setup**:
   - Add Gazebo plugins to your URDF
   - Create a launch file to spawn the robot in Gazebo
   - Test the robot's response to applied forces

5. **Advanced Modeling**:
   - Add sensors (IMU, cameras) to your humanoid robot
   - Include transmission elements for actuator control
   - Model a simplified hand with grasp capabilities

## Summary

URDF is a fundamental component in robotics, especially for humanoid robots with complex kinematic structures. Understanding how to properly construct URDF files with accurate inertial properties, realistic joint limits, and appropriate visual/collision geometries is crucial for successful robot simulation and control.

The use of Xacro macros simplifies the creation of repetitive structures common in humanoid robots, while proper Gazebo integration enables effective simulation and testing. Following best practices for URDF construction ensures stable simulations and accurate robot behavior.

In the next chapter, we'll explore launch files and parameters, which are essential for configuring and launching complex robotic systems like the humanoid robot we've modeled.