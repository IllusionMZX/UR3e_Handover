<div align="center">

# 🤖 UR3e Human-to-Robot Handover Based on Stereo Vision 👁️

[🇬🇧 English](#english) | [🇨🇳 中文](#chinese)

</div>

<span id="english"></span>

## 🇬🇧 English

**Note:** This project is currently under development and will continue to be updated.

### ✨ Implemented Features
- 🤖 UR3e robot driver launch and control.
- 📷 RealSense D435i camera startup.
- 👁️ Publishing hand-eye calibration results.
- 🤏 Robotiq 2F gripper ROS node control.
- 📦 Vision-based pick and place (HSV color detection).
- 🎯 Real-time object tracking (HSV).
- 📍 Moving the robot to specific coordinates.

### 1. 🛠️ Environment Configuration

> **❗ Note:** All packages were downloaded and installed in an open network environment, and the user mode during configuration was `root`. If you encounter issues with `git`, `apt`, or `rosdep` downloading, updating, or installing, please change sources, use a proxy, or search for relevant network solutions.

> **❗ Note:** During robot movement, always pay attention to whether the connecting cables are tangled! Always check if the robot arm is colliding with other objects! If any of the above situations occur, immediately press the Emergency Stop button on the teach pendant! Ensure the safety of the robot arm!

- **🤖 Robot:** UR3e
- **🖥️ OS:** Ubuntu 22.04.5 LTS (Physical Machine, not Virtual Machine)
- **🐢 ROS Version:** ROS2 Humble
- **🎮 GPU:** NVIDIA 3050 Ti Mobile
- **🚗 GPU Driver:** NVIDIA driver metapackage from nvidia-driver-570
- **🧠 CPU:** 12th Gen Intel i7-12700H
- **💻 IDE:** VS Code
- **🐍 Python:** 3.10.12

### 2. 🚀 Installation and Usage

For a detailed tutorial, please refer to: [UR3e Pick and Place User Manual](https://protective-calendula-c55.notion.site/UR3e-Pick-and-Place-User-Manual-2c3aa15567ec80c6ac5ad8a5c6b1374b)

```bash
sudo su

mkdir -p ~/workspace/

cd ~/workspace/

# Recursively clone the repository, including all submodules
git clone --recurse-submodules https://github.com/IllusionMZX/UR3e_Handover_Vision

# Enter the workspace source directory
cd ./src/

# Install dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build

# Note: Open a new terminal for each ROS command. 
# After opening a terminal, navigate to the workspace directory and run the source command to reload the ROS2 environment.
source install/setup.bash

# Launch ur_robot_driver
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=192.168.1.10

# Launch RealSense D435i
ros2 launch realsense2_camera rs_launch.py

# Publish hand-eye calibration results
ros2 launch calibration_result camerapose3.launch.py

# Start Robotiq gripper ROS node
ros2 run robotiq_2f_urcap_adapter robotiq_2f_adapter_node.py --ros-args -p robot_ip:=192.168.1.10

# Run pick and place program
ros2 run vision_pick_and_place hsv_pick_and_place

# Run real-time tracking program
ros2 run vision_pick_and_place hsv_tracking

# Run move to specific place program
python3 ./src/vision_pick_and_place/move_to_specific_place.py
```

---

<span id="chinese"></span>

## 🇨🇳 中文

**注：** 本项目目前正在开发中，后续将继续更新。

### ✨ 已实现功能
- 🤖 启动 UR3e 机械臂驱动。
- 📷 启动 RealSense D435i 相机。
- 👁️ 发布手眼标定结果。
- 🤏 启动 Robotiq 2F 夹爪 ROS 节点。
- 📦 运行抓取、放置程序 (基于 HSV)。
- 🎯 运行实时跟踪程序 (基于 HSV)。
- 📍 运行移动到指定位置程序。

### 1. 🛠️ 环境配置

> **❗ 注：** 所有功能包均在外网环境下载安装，且环境配置过程中用户模式为 `root` 模式。若遇到 `git`、`apt`、`rosdep` 无法下载、更新、安装等情况，自行换源、更换代理节点或查找相关网络资料。

> **❗ 注：** 机械臂运动过程中，时刻注意连接线是否发生缠绕！时刻注意机械臂是否碰撞到其他物体！遇到上述情况，立即按下示教器上紧急停止按钮！确保机械臂安全！

- **🤖 Robot:** UR3e
- **🖥️ OS:** Ubuntu 22.04.5 LTS (物理机，非虚拟机)
- **🐢 ROS Version:** ROS2 Humble
- **🎮 GPU:** NVIDIA 3050 Ti Mobile
- **🚗 GPU Driver:** NVIDIA driver metapackage from nvidia-driver-570
- **🧠 CPU:** 12th Gen Intel i7-12700H
- **💻 IDE:** VS Code
- **🐍 Python:** 3.10.12

### 2. 🚀 安装与使用

详细教程请参考：[UR3e Pick and Place User Manual](https://protective-calendula-c55.notion.site/UR3e-Pick-and-Place-User-Manual-2c3aa15567ec80c6ac5ad8a5c6b1374b)

```bash
sudo su

mkdir -p ~/workspace/

cd ~/workspace/

# 递归clone仓库，包含所有submodule子模块
git clone --recurse-submodules https://github.com/IllusionMZX/UR3e_Handover_Vision

# 进入工作空间
cd ./src/

# 下载依赖包
rosdep install --from-paths src --ignore-src -r -y

# 构建
colcon build

# 注意：每运行一个ros命令需打开一个新终端，打开终端后都要进入工作空间目录运行该命令重新加载ROS2环境
source install/setup.bash

# 启动ur_robot_driver
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=192.168.1.10

# 启动realsense d435i
ros2 launch realsense2_camera rs_launch.py

# 发布手眼标定结果
ros2 launch calibration_result camerapose3.launch.py

# 启动夹爪ros节点
ros2 run robotiq_2f_urcap_adapter robotiq_2f_adapter_node.py --ros-args -p robot_ip:=192.168.1.10

# 运行抓取、放置程序
ros2 run vision_pick_and_place hsv_pick_and_place

# 运行实时跟踪程序
ros2 run vision_pick_and_place hsv_tracking

# 运行移动到指定位置程序
python3 ./src/vision_pick_and_place/move_to_specific_place.py
```