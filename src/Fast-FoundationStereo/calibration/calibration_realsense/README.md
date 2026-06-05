# RealSense 双相机标定与测试流程

由于我们需要使用两台独立的 RealSense D435(或其他型号) 的 RGB 颜色流来构成一个新的“双目立体相机”，而不是使用它们出厂自带的 IR 深度流，因此我们**必须对这两个独立相机进行重新联合标定（ Stereo Calibration ）**。这能够告诉系统它们之间的相对旋转矩阵与平移量（基线）。

### 准备工作

请确保已经安装并准备好了一张 A4 纸大小的 **ChArUco 标定板**。
我们默认预设的标定板参数如下，如果您打印的比例略有不同，请**务必修改**以下代码中的参数 (`collect_cali_rs.py` 和 `calibrate_rs.py`):
```python
cv2.aruco.CharucoBoard((7, 5), 0.030, 0.0225, aruco_dict)
```

### 第一步：收集标定图像

1. 将两个 RealSense 摄像头固定好（一旦固定，就不允许发生相对移动，否则需要重新标定）。
2. 运行采集脚本：`python calibration_realsense/collect_cali_rs.py`
3. 移动标定板，从不同角度、不同距离覆盖相机画面的全部区域（角落、边缘、中心），尤其注意标定板要在左眼和右眼中**同时全部可见**。
4. 按下**空格键**捕获图像。请尽量收集大约 20 到 30 对包含不同视角的图像。
5. 按 **q** 退出采集，图像会保存在 `calibration_realsense/calib_imgs/` 。

### 第二步：计算内参、外参及极线校正矩阵并导出

1. 运行标定脚本：`python calibration_realsense/calibrate_rs.py`
2. 脚本将开始分析采集到的所有左右图像。
3. 执行成功后，将在当前目录生成一个 `stereo_calib_rs.yaml` 文件。这个文件包含推导立体深度所需的基线、校正映射等核心数据。

### 第三步：运行双 RealSense FFS 推理实时预览

完成标定并拥有了 `stereo_calib_rs.yaml` 后，即可直接运行我们为您准备好的推理 Demo 脚本：
```bash
python ffsd_demos/dual_realsense_ffs_realtime.py
```
这会调用 FFS 神经网络生成左右相机的深度视差，并在 Open3D 空间内将它们与颜色纹理结合展示。