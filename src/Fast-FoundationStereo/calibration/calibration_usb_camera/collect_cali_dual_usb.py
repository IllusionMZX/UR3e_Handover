import cv2
import os
import numpy as np

# ====== 参数配置 ======
# 统一直接使用 1280x720 原生 16:9 采集，避免 4:3 硬件强行挤压变形
LEFT_W = 1280
LEFT_H = 720

RIGHT_W = 1280
RIGHT_H = 720

# 拼图与保存时的最终统一目标分辨率
FINAL_W = 1280
FINAL_H = 720

# ChArUco 标定板参数
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
charuco_board = cv2.aruco.CharucoBoard((7, 5), 0.030, 0.0225, aruco_dict)
detector = cv2.aruco.CharucoDetector(charuco_board)

# ====== 目录初始化 ======
CALIB_DIR = os.path.dirname(os.path.realpath(__file__))
CALIB_IMG_DIR = os.path.join(CALIB_DIR, 'calib_imgs')
os.makedirs(os.path.join(CALIB_IMG_DIR, 'left'), exist_ok=True)
os.makedirs(os.path.join(CALIB_IMG_DIR, 'right'), exist_ok=True)

# ====== 摄像头初始化与自动识别 ======
def init_camera(dev_id):
    cap = cv2.VideoCapture(dev_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, LEFT_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LEFT_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

print("正在扫描可用的 USB 相机设备 (测试 /dev/video0 到 10)...")
valid_devs = []
for i in range(12):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, _ = cap.read()
        if ret:
            valid_devs.append(i)
    cap.release()
    if len(valid_devs) == 2:
        break

if len(valid_devs) < 2:
    print(f"错误: 仅找到 {len(valid_devs)} 台正常输出画面的相机: {valid_devs}，请检查连接或USB带宽。")
    exit(1)

dev_0, dev_1 = valid_devs[0], valid_devs[1]
print(f"发现可用相机: /dev/video{dev_0} 和 /dev/video{dev_1}")

cap0 = init_camera(dev_0)
cap1 = init_camera(dev_1)

print("正在自动识别左右相机... 请将标定板放在两台相机共同视野内！")
DEV_LEFT = None
DEV_RIGHT = None
cap_left, cap_right = None, None

while True:
    ret0 = cap0.grab()
    ret1 = cap1.grab()
    
    if not (ret0 and ret1):
        print(f"[警告] 同步拉取图像失败 (ret0={ret0}, ret1={ret1})，请检查 USB 带宽是否足够 (尝试分别插在主板不同 USB 控制器接口上)!")
        cap0.release()
        cap1.release()
        exit(1)
    
    _, img0 = cap0.retrieve()
    _, img1 = cap1.retrieve()

    c0, ids0, _, _ = detector.detectBoard(img0)
    c1, ids1, _, _ = detector.detectBoard(img1)

    if c0 is not None and c1 is not None and len(c0) > 4 and len(c1) > 4:
        # 找到公共检测到的角点 ID
        common_ids = np.intersect1d(ids0, ids1)
        if len(common_ids) > 4:
            x0_mean = np.mean([c0[i][0][0] for i in range(len(ids0)) if ids0[i] in common_ids])
            x1_mean = np.mean([c1[i][0][0] for i in range(len(ids1)) if ids1[i] in common_ids])
            
            if x0_mean > x1_mean:
                DEV_LEFT, DEV_RIGHT = dev_0, dev_1
                cap_left, cap_right = cap0, cap1
            else:
                DEV_LEFT, DEV_RIGHT = dev_1, dev_0
                cap_left, cap_right = cap1, cap0
            
            print(f"成功识别！左相机: /dev/video{DEV_LEFT}，右相机: /dev/video{DEV_RIGHT}")
            break

        # 缩放至小尺寸用于识别预览，保持原有长宽比避免预览拉伸
        h0, w0 = img0.shape[:2]
        prev0 = cv2.resize(img0, (640, int(640 * h0 / w0)))
        h1, w1 = img1.shape[:2]
        prev1 = cv2.resize(img1, (640, int(640 * h1 / w1)))
        cv2.imshow("Auto-detecting Left/Right... Show the board!", cv2.hconcat([prev0, prev1]))
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()

if DEV_LEFT is None:
    print("未能识别左右相机即退出。")
    cap0.release()
    cap1.release()
    exit(1)

count = 0
print("\n[ 操作提示 ]")
print("  按下 空格键: 保存当前左右图像 (仅当左眼检测到标定板时)")
print("  按下 'q' 键: 退出录制")

def center_crop_and_resize(img, trg_w, trg_h):
    # 不再进行画面裁剪，保留完整摄像头原始视角
    return cv2.resize(img, (trg_w, trg_h))

while True:
    # 尽可能同步地读取两台相机的画面 (先 grab() 后 retrieve()，以减少时间差)
    ret_l = cap_left.grab()
    ret_r = cap_right.grab()
    
    if not ret_l and not ret_r:
        print("[Error] Both cameras completely failed to grab.")
        break
        
    _, img_left = cap_left.retrieve() if ret_l else (False, np.zeros((LEFT_H, LEFT_W, 3), dtype=np.uint8))
    _, img_right = cap_right.retrieve() if ret_r else (False, np.zeros((RIGHT_H, RIGHT_W, 3), dtype=np.uint8))
    
    if not ret_l or not ret_r:
        print(f"[Warning] Frame drop - Left: {ret_l}, Right: {ret_r}")
        # 即使有一边失败，我们也允许显示另外一半或者黑屏，而不是 continue 死循环
    
    # 根据实际画面的长宽比例，智能居中裁剪后再缩放，避免拉伸变形
    img_left = center_crop_and_resize(img_left, FINAL_W, FINAL_H)
    img_right = center_crop_and_resize(img_right, FINAL_W, FINAL_H)

    # 检测并绘制 ChArUco (基于左图)
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
    
    preview = cv2.hconcat([vis_left, img_right])
    
    # 640x480 双屏拼接后分辨率为 1280x480，屏幕可直接完整显示
    preview_show = preview
    cv2.putText(preview_show, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.imshow('Dual USB Cameras', preview_show)

    key = cv2.waitKey(1)
    if key == ord(' '):
        if detected:
            cv2.imwrite(os.path.join(CALIB_IMG_DIR, f'left/{count:03d}.png'), img_left)
            cv2.imwrite(os.path.join(CALIB_IMG_DIR, f'right/{count:03d}.png'), img_right)
            print(f"  已保存 #{count:03d} (检测到 {len(charuco_corners)} 个内部角点)")
            count += 1
        else:
            print("  未检测到足够的 ChArUco 角点，跳过保存。")
    elif key == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
