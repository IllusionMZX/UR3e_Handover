# stereo_calibrate_charuco.py
import cv2
import numpy as np
import glob
import yaml
import os

# Calibration directory & repo root
CALIB_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(CALIB_DIR)

# ── Parameters ──────────────────────────────────
SQUARES_X   = 7
SQUARES_Y   = 5
SQUARE_SIZE = 0.030   # Measured value, in meters
MARKER_SIZE = 0.022  # 修改为标定板使用的准确测量值 0.0225
# ─────────────────────────────────────────────────

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_SIZE,
    MARKER_SIZE,
    dictionary
)
detector = cv2.aruco.CharucoDetector(board)

objpoints   = []
imgpoints_l = []
imgpoints_r = []

left_imgs  = sorted(glob.glob(os.path.join(CALIB_DIR, 'calib_imgs/left/*.png')))
right_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, 'calib_imgs/right/*.png')))

valid_count = 0
for i, (lp, rp) in enumerate(zip(left_imgs, right_imgs)):
    img_l = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)

    # Detect ChArUco corners
    corners_l, ids_l, _, _ = detector.detectBoard(img_l)
    corners_r, ids_r, _, _ = detector.detectBoard(img_r)

    if corners_l is None or corners_r is None:
        print(f"  Pair {i:2d} - detection failed")
        continue

    # Find corners visible in both left and right images
    ids_l_flat = ids_l.flatten()
    ids_r_flat = ids_r.flatten()
    common_ids = np.intersect1d(ids_l_flat, ids_r_flat)

    if len(common_ids) < 6:
        print(f"  Pair {i:2d} - too few common corners ({len(common_ids)})")
        continue

    # Use matchImagePoints to get 3D coordinates securely
    obj_points_l, img_points_l = board.matchImagePoints(corners_l, ids_l)
    obj_points_r, img_points_r = board.matchImagePoints(corners_r, ids_r)
    
    if obj_points_l is None or obj_points_r is None:
        continue
        
    match_obj = []
    match_l = []
    match_r = []
    
    # Exact point matching based on physical 3D coordinates (just like calibrate_rs.py)
    for p_obj, p_img in zip(obj_points_l, img_points_l):
        key = str(np.round(p_obj, 5))
        for p_obj_r, p_img_r in zip(obj_points_r, img_points_r):
            if str(np.round(p_obj_r, 5)) == key:
                match_obj.append(p_obj)
                match_l.append(p_img)
                match_r.append(p_img_r)
                break
    
    if len(match_obj) < 6:
        print(f"  Pair {i:2d} - matching valid object corners failed")
        continue

    objpoints.append(np.array(match_obj, dtype=np.float32))
    imgpoints_l.append(np.array(match_l, dtype=np.float32))
    imgpoints_r.append(np.array(match_r, dtype=np.float32))
    valid_count += 1
    print(f"  Pair {i:2d} - {len(match_obj)} common corners")

print(f"\nValid image pairs: {valid_count}")
assert valid_count >= 15, "Too few valid images, recollect"

H, W = img_l.shape
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

# Monocular calibration
_, K_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, (W,H), None, None)
_, K_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, (W,H), None, None)

print(f"Monocular Left fx: {K_l[0,0]:.2f}, Right fx: {K_r[0,0]:.2f}")

# Stereo calibration
rms, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    K_l, dist_l, K_r, dist_r,
    (W, H),
    flags=cv2.CALIB_FIX_INTRINSIC,
    criteria=criteria
)

# Compute rectification maps

R_l, R_r, P_l, P_r, Q, validROI_l, validROI_r = cv2.stereoRectify(
    K_l, dist_l, K_r, dist_r, (W,H), R, T, alpha=0)

baseline = abs(P_r[0, 3] / P_r[0, 0])
print(f"\nRMS: {rms:.4f}")
print(f"Physical Baseline: {baseline*1000:.2f}mm")

# Save
calib = {
    'baseline': float(baseline),
    'K_l': K_l.tolist(), 'dist_l': dist_l.tolist(),
    'K_r': K_r.tolist(), 'dist_r': dist_r.tolist(),
    'R': R.tolist(), 'T': T.tolist(),
    'R_l': R_l.tolist(), 'R_r': R_r.tolist(),
    'P_l': P_l.tolist(), 'P_r': P_r.tolist(),
}
with open(os.path.join(CALIB_DIR, 'stereo_calib.yaml'), 'w') as f:
    yaml.dump(calib, f)

K_rect = P_l[:3, :3]
with open(os.path.join(CALIB_DIR, 'K_custom.txt'), 'w') as f:
    f.write(' '.join([f'{v:.6f}' for v in K_rect.flatten()]) + '\n')
    f.write(f'{baseline:.6f}\n')

print("Done")
