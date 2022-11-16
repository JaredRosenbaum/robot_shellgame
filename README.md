# MoveIt Packages with Python Scripts Created for ME 396P Final Presentation, Fall 2022, Team G09

## Contents
1. **ur_moveit_config**
    * The ur_moveit_config package contains the simplest possible MoveIt configuration for a Universal Robots UR3 manipulator developed using the MoveIt Setup Assistant. It depends on UR-provided packages which can be found at the [Universal Robots ROS Drivers](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver) page.
2. **ur_scripts**
    * The ur_scripts package contains the Python code used to make the robot compute and execute operations. In order for the scripts to properly execute, one must be using the same experimental setup (1 UR3 robot arm, 1 Intel RealSense D435i camera + end effector, 3 Red Shells, 1 green marking on a Shell)
---

## Requirements, Dependencies, and Building
These packages are built and tested on a system running ROS1 noetic on Ubuntu 20.04. Users are assumed to already have ROS noetic installed on a machine running Ubuntu 20.04 to execute this demonstration. Details of ROS installation can be found on the [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials) webpage.

Use of these packages in a non-simulated environment requires the use of the official [Universal Robots ROS Drivers](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver).   
1. Create a Catkin workspace:
```console
mkdir -p catkin_ws/src && cd catkin_ws
```
2. Clone the contents of this repository:
```console
git clone https://github.com/steven-swanbeck/ur3_calculator.git src/lightning_talk
```
3. Clone the UR Robots ROS Driver:
```console
git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git src/Universal_Robots_ROS_Driver
```
4. And the associated description packages:
```console
git clone -b melodic-devel-staging https://github.com/ros-industrial/universal_robot.git src/universal_robot
```
5. Install all package dependencies:
```console
sudo apt update -qq
```
```console
rosdep update
```
```console
rosdep install --from-paths src --ignore-src -r -y
```
6. Make the workspace:
```console
catkin_make
```
7. Source the workspace:
```console
source devel/setup.bash
```
---








