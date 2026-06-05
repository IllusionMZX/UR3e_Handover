# collect_cali_rs.py
import pyrealsense2 as rs
import cv2
import numpy as np
import os

# ====== 配置参数 ======
IMG_WIDTH = 640
IMG_HEIGHT = 480
FPS = 30

# 使用你之前在 A4 纸上生成并确认打印准确的尺寸参数
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
charuco_board = cv2.aruco.CharucoBoard((7, 5), 0.030, 0.0225, aruco_dict)
detector = cv2.aruco.CharucoDetector(charuco_board)

# ====== 目录初始化 ======
CALIB_DIR = os.path.dirname(os.path.realpath(__file__))
CALIB_IMG_DIR = os.path.join(CALIB_DIR, 'calib_imgs')
os.makedirs(os.path.join(CALIB_IMG_DIR, 'left'), exist_ok=True)
os.makedirs(os.path.join(CALIB_IMG_DIR, 'right'), exist_ok=True)

# ====== 初始化两个 RealSense ======
ctx = rs.context()
devices = ctx.query_devices()
if len(devices) < 2:
    raise RuntimeError("需要连接至少两台 RealSense 设备！")

# 获取序列号
serial_0 = devices[0].get_info(rs.camera_info.serial_number)
serial_1 = devices[1].get_info(rs.camera_info.serial_number)

print("正在自动识别左右相机... 请将标定板放在两台相机共同视野内！")

# 临时启动流用于识别
pipe0 = rs.pipeline()
cfg0 = rs.config()
cfg0.enable_device(serial_0)
cfg0.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, FPS)
pipe0.start(cfg0)

pipe1 = rs.pipeline()
cfg1 = rs.config()
cfg1.enable_device(serial_1)
cfg1.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, FPS)
pipe1.start(cfg1)

serial_left = None
serial_right = None

try:
    while True:
        f0 = pipe0.wait_for_frames()
        f1 = pipe1.wait_for_frames()
        
        img0 = np.asanyarray(f0.get_color_frame().get_data())
        img1 = np.asanyarray(f1.get_color_frame().get_data())
        
        # 检测两图中的角点
        c0, ids0, _, _ = detector.detectBoard(img0)
        c1, ids1, _, _ = detector.detectBoard(img1)
        
        if c0 is not None and c1 is not None and len(c0) > 4 and len(c1) > 4:
            # 找到公共检测到的角点 ID
            common_ids = np.intersect1d(ids0, ids1)
            if len(common_ids) > 4:
                # 提取共同角点的平均 x 坐标
                x0_mean = np.mean([c0[i][0][0] for i in range(len(ids0)) if ids0[i] in common_ids])
                x1_mean = np.mean([c1[i][0][0] for i in range(len(ids1)) if ids1[i] in common_ids])
                
                # 左相机的物理位置偏左（x较小），但它看到的物体偏右（图像上的 x 坐标较大）
                if x0_mean > x1_mean:
                    serial_left, serial_right = serial_0, serial_1
                else:
                    serial_left, serial_right = serial_1, serial_0
                
                print("成功识别！")
                break
        
        cv2.imshow("Auto-detecting Left/Right... Show the board!", cv2.hconcat([img0, img1]))
        cv2.waitKey(1)
finally:
    pipe0.stop()
    pipe1.stop()
    cv2.destroyAllWindows()

print(f"检测到 Left 相机: {serial_left}")
print(f"检测到 Right 相机: {serial_right}")

# 把序列号保存下来，供后续推理程序读取
import yaml
sn_file = os.path.join(CALIB_DIR, "stereo_sn_rs.yaml")
with open(sn_file, "w") as f:
    yaml.dump({"serial_left": serial_left, "serial_right": serial_right}, f)
print(f"已将相机序列号存入: {sn_file}")

# 配置左相机
pipeline_left = rs.pipeline()
config_left = rs.config()
config_left.enable_device(serial_left)
config_left.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, FPS)
pipeline_left.start(config_left)

# 配置右相机
pipeline_right = rs.pipeline()
config_right = rs.config()
config_right.enable_device(serial_right)
config_right.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, FPS)
pipeline_right.start(config_right)

count = 0
print("\n[ 操作提示 ]")
print("  按下 空格键: 保存当前左右图像 (仅当检测到标定板时)")
print("  按下 'q' 键: 退出录制")

try:
    while True:
        # 获取帧
        frames_left = pipeline_left.wait_for_frames()
        frames_right = pipeline_right.wait_for_frames()
        
        color_frame_left = frames_left.get_color_frame()
        color_frame_right = frames_right.get_color_frame()
        
        if not color_frame_left or not color_frame_right:
            continue
            
        # 转为 numpy 数组
        img_left = np.asanyarray(color_frame_left.get_data())
        img_right = np.asanyarray(color_frame_right.get_data())

        # 检测角点 (仅用左图做可视化反馈)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img_left)
        vis_left = img_left.copy()
        
        detected = False
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis_left, marker_corners, marker_ids)
            if charuco_corners is not None and len(charuco_corners) >= 4:
                cv2.aruco.drawDetectedCornersCharuco(vis_left, charuco_corners, charuco_ids, (0, 0, 255))
                detected = True

        status_color = (0, 255, 0) if detected else (0, 0, 255)
        status_text = f"Saved: {count} | ChArUco: {'OK' if detected else 'N/A'}"
        
        # 拼接图像用于预览
        preview = cv2.hconcat([vis_left, img_right])
        cv2.putText(preview, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.imshow('Dual RealSense Catch', preview)

        key = cv2.waitKey(1)
        if key == ord(' '):
            if detected:
                cv2.imwrite(os.path.join(CALIB_IMG_DIR, f'left/{count:03d}.png'), img_left)
                cv2.imwrite(os.path.join(CALIB_IMG_DIR, f'right/{count:03d}.png'), img_right)
                print(f"  已保存 #{count:03d} (检测到 {len(charuco_corners)} 个内部角点)")
                count += 1
            else:
                print("  未检测到足够的 ChArUco 角点，跳过保存。请移动标定板使其清晰可见。")
        elif key == ord('q'):
            break

finally:
    pipeline_left.stop()
    pipeline_right.stop()
    cv2.destroyAllWindows()
    print("标定图像采集结束。")
