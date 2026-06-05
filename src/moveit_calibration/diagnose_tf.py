#!/usr/bin/env python3
"""
TF树诊断工具 - 用于检查手眼标定和机器人TF链是否正确
"""
import rclpy
from rclpy.node import Node
import tf2_ros
from tf2_ros import TransformException
import sys

class TFDiagnose(Node):
    def __init__(self):
        super().__init__('tf_diagnose')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 等待TF数据
        self.get_logger().info('等待TF数据...')
        import time
        time.sleep(2.0)
        
        self.check_tf_tree()
        
    def check_tf_tree(self):
        """检查关键的TF变换链"""
        print("\n" + "="*60)
        print("TF树诊断报告")
        print("="*60)
        
        # 定义需要检查的关键frame
        frames_to_check = [
            'base_link',
            'base_link_inertia', 
            'wrist_3_link',
            'tool0',
            'camera_color_optical_frame',
            'camera_link',
            'camera_color_frame'
        ]
        
        # 1. 检查frame是否存在
        print("\n1. 检查关键Frame是否存在:")
        print("-" * 60)
        existing_frames = []
        for frame in frames_to_check:
            try:
                # 尝试查找该frame
                if self.tf_buffer._frameExists(frame):
                    print(f"  ✓ {frame:30s} [存在]")
                    existing_frames.append(frame)
                else:
                    print(f"  ✗ {frame:30s} [不存在]")
            except:
                print(f"  ✗ {frame:30s} [不存在]")
        
        # 2. 检查关键TF链
        print("\n2. 检查TF变换链:")
        print("-" * 60)
        
        critical_transforms = [
            ('base_link', 'wrist_3_link', '机器人基座 -> 手腕3'),
            ('wrist_3_link', 'camera_color_optical_frame', '手腕3 -> 相机光学中心(手眼标定)'),
            ('base_link', 'camera_color_optical_frame', '机器人基座 -> 相机(完整链)'),
            ('base_link', 'tool0', '机器人基座 -> 末端执行器'),
        ]
        
        for parent, child, desc in critical_transforms:
            try:
                if self.tf_buffer.can_transform(parent, child, rclpy.time.Time()):
                    trans = self.tf_buffer.lookup_transform(parent, child, rclpy.time.Time())
                    t = trans.transform.translation
                    r = trans.transform.rotation
                    print(f"  ✓ {desc}")
                    print(f"    {parent} -> {child}")
                    print(f"    平移: [{t.x:.4f}, {t.y:.4f}, {t.z:.4f}]")
                    print(f"    旋转: [{r.x:.4f}, {r.y:.4f}, {r.z:.4f}, {r.w:.4f}]")
                else:
                    print(f"  ✗ {desc}")
                    print(f"    {parent} -> {child} [变换不可用]")
            except TransformException as e:
                print(f"  ✗ {desc}")
                print(f"    {parent} -> {child}")
                print(f"    错误: {e}")
        
        # 3. 检查相机相关的可能frame名称
        print("\n3. 搜索相机相关的Frame:")
        print("-" * 60)
        possible_camera_frames = [
            'camera_link',
            'camera_color_frame',
            'camera_color_optical_frame',
            'camera_depth_frame',
            'camera_depth_optical_frame',
            'camera_aligned_depth_to_color_frame'
        ]
        
        found_camera_frames = []
        for frame in possible_camera_frames:
            try:
                if self.tf_buffer._frameExists(frame):
                    print(f"  ✓ {frame}")
                    found_camera_frames.append(frame)
            except:
                pass
        
        if not found_camera_frames:
            print("  ✗ 未找到相机相关的frame!")
            print("  提示: 请确认RealSense驱动已启动")
        
        # 4. 诊断建议
        print("\n4. 诊断结果和建议:")
        print("-" * 60)
        
        issues = []
        
        # 检查手眼标定TF是否发布
        if 'camera_color_optical_frame' not in existing_frames:
            issues.append("❌ 相机光学frame不存在!")
            issues.append("   解决方法: 启动RealSense驱动")
            issues.append("   命令: ros2 launch realsense2_camera rs_launch.py")
        
        try:
            if not self.tf_buffer.can_transform('wrist_3_link', 'camera_color_optical_frame', rclpy.time.Time()):
                issues.append("❌ 手眼标定TF未发布 (wrist_3_link -> camera_color_optical_frame)")
                issues.append("   解决方法: 启动手眼标定launch文件")
                issues.append("   命令: ros2 launch /root/workspace/ros_ur_driver/src/moveit_calibration/camerapose2.launch.py")
        except:
            pass
        
        try:
            if not self.tf_buffer.can_transform('base_link', 'camera_color_optical_frame', rclpy.time.Time()):
                issues.append("❌ 完整TF链断裂 (base_link无法到达camera)")
                issues.append("   检查: 机器人驱动是否运行, TF链是否完整")
        except:
            pass
        
        if not issues:
            print("  ✓ 所有关键TF链都正常!")
            print("  系统可以开始跟踪木块")
        else:
            for issue in issues:
                print(f"  {issue}")
        
        print("\n" + "="*60)
        print("诊断完成")
        print("="*60 + "\n")

def main(args=None):
    rclpy.init(args=args)
    node = TFDiagnose()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
