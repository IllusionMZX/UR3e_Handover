import cv2
import numpy as np
import glob
import os
import yaml

CALIB_DIR = os.path.dirname(os.path.realpath(__file__))
# 修改此处的尺寸以匹配你的打印版本
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
charuco_board = cv2.aruco.CharucoBoard((7, 5), 0.030, 0.022, aruco_dict)

def calibrate_single_camera(img_dir):
    all_charuco_corners = []
    all_charuco_ids = []
    all_obj_points = []
    all_img_points = []
    
    images = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    assert len(images) > 0, f"没有在 {img_dir} 找到图片。"
    
    img_shape = None
    detector = cv2.aruco.CharucoDetector(charuco_board)
    
    valid_images = []
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]
            
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        if charuco_corners is not None and len(charuco_corners) > 6:
            # 对应的世界坐标
            obj_points, img_points = charuco_board.matchImagePoints(charuco_corners, charuco_ids)
            if obj_points is not None and len(obj_points) > 0:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                all_obj_points.append(obj_points)
                all_img_points.append(img_points)
                valid_images.append(fname)
                
    # 单相机内参标定
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(all_obj_points, all_img_points, img_shape, None, None)
    return ret, K, dist, valid_images, all_obj_points, all_img_points

def main():
    print("开始标定左相机...")
    ret_l, K_l, dist_l, valid_l, obj_pts_l, img_pts_l = calibrate_single_camera(os.path.join(CALIB_DIR, 'calib_imgs/left'))
    print(f"左相机标定完成, RMS误差: {ret_l:.3f}")

    print("开始标定右相机...")    
    ret_r, K_r, dist_r, valid_r, obj_pts_r, img_pts_r = calibrate_single_camera(os.path.join(CALIB_DIR, 'calib_imgs/right'))
    print(f"右相机标定完成, RMS误差: {ret_r:.3f}")

    # 取共用成功识别到特征点的图像帧进行双目标定
    common_images = list(set([os.path.basename(f) for f in valid_l]) & set([os.path.basename(f) for f in valid_r]))
    common_images.sort()
    
    sync_obj_pts = []
    sync_img_pts_l = []
    sync_img_pts_r = []
    
    img_shape = None
    for basename in common_images:
        gray_l = cv2.cvtColor(cv2.imread(os.path.join(CALIB_DIR, 'calib_imgs/left', basename)), cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(cv2.imread(os.path.join(CALIB_DIR, 'calib_imgs/right', basename)), cv2.COLOR_BGR2GRAY)
        img_shape = gray_l.shape[::-1]
        
        detector = cv2.aruco.CharucoDetector(charuco_board)
        corners_l, ids_l, _, _ = detector.detectBoard(gray_l)
        corners_r, ids_r, _, _ = detector.detectBoard(gray_r)
        
        if corners_l is None or corners_r is None: continue
        
        # 寻找左右图都检测到的相同角点 ID
        common_ids = set(ids_l.flatten()) & set(ids_r.flatten())
        if len(common_ids) < 6: continue
        
        match_obj = []
        match_l = []
        match_r = []
        
        obj_points_l, img_points_l = charuco_board.matchImagePoints(corners_l, ids_l)
        obj_points_r, img_points_r = charuco_board.matchImagePoints(corners_r, ids_r)
        
        for p_obj, p_img in zip(obj_points_l, img_points_l):
            # 将角点物理坐标转为字符串哈希用于匹配
            key = str(np.round(p_obj, 5))
            # 扫描右图里是否有匹配的物理坐标点
            for p_obj_r, p_img_r in zip(obj_points_r, img_points_r):
                if str(np.round(p_obj_r, 5)) == key:
                    match_obj.append(p_obj)
                    match_l.append(p_img)
                    match_r.append(p_img_r)
                    break
        
        if len(match_obj) >= 6:
            sync_obj_pts.append(np.array(match_obj))
            sync_img_pts_l.append(np.array(match_l))
            sync_img_pts_r.append(np.array(match_r))

    print(f"找到 {len(sync_obj_pts)} 对有效双目图像，进行双目标定...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    ret_stereo, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        sync_obj_pts, sync_img_pts_l, sync_img_pts_r,
        K_l, dist_l, K_r, dist_r,
        img_shape, criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
    )
    print(f"双目标定完成, RMS误差: {ret_stereo:.3f} (基线长度: {np.linalg.norm(T):.4f}m)")

    # 极线校正 (Rectification)
    R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        K_l, dist_l, K_r, dist_r, img_shape, R, T, alpha=0)

    # 导出到 YAML 用于推理脚本
    calib_data = {
        'K_l': K_l.tolist(), 'dist_l': dist_l.tolist(),
        'K_r': K_r.tolist(), 'dist_r': dist_r.tolist(),
        'R_l': R_l.tolist(), 'R_r': R_r.tolist(),
        'P_l': P_l.tolist(), 'P_r': P_r.tolist(),
        'baseline': float(abs(P_r[0, 3] / P_r[0, 0])) # px的水平偏移除以焦距得到基线 (米)
    }

    out_file = os.path.join(CALIB_DIR, 'stereo_calib_rs.yaml')
    with open(out_file, 'w') as f:
        yaml.dump(calib_data, f, default_flow_style=False)
    print(f"标定参数已保存至: {out_file}")

if __name__ == '__main__':
    main()
