import cv2.aruco as aruco
import numpy as np
import torch
import os
import cv2
from sklearn.linear_model import LinearRegression
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
parameters = aruco.DetectorParameters_create()
camera_matrix = np.array([
    [580.77518, 0.0, 724.75002],
    [0.0, 580.77518, 570.98956],
    [0.0, 0.0, 1.0]
])
distortion_coeffs = np.array([
    [0.927077],
    [0.141438],
    [0.000196],
    [-8.7e-05],
    [0.001695],
    [1.257216],
    [0.354688],
    [0.015954]
])


def get_depth_map(model: DepthAnythingV2,
                     img_path: str,
                     save_dir: str = None) -> tuple[np.ndarray, float]:
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # model inference
    depth = model.infer_image(img)
    # getting distance from camera to aruco markers
    corners, ids, rej = aruco.detectMarkers(
        img_gray, aruco_dict, parameters=parameters
    )
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
        corners, 0.08, camera_matrix, distortion_coeffs
    )
    z_vals = []
    depth_vals = []
    for ind, cur_tvec in enumerate(tvec):
        z = abs(cur_tvec[0][2])  # getting z coordinate, abs just in case
        corner = corners[ind][0]
        tl, br = corner[0], corner[2]
        center = (int((tl[1] + br[1]) / 2), int((tl[0] + br[0]) / 2))
        z_vals.append(z)
        depth_vals.append([depth[center[0]][center[1]]])

    # scaling the depth map
    if len(z_vals) == 0:
        pass
    elif len(z_vals) == 1:
        depth = depth * (z_vals[0] / depth_vals[0])
    else:
        lin = LinearRegression()
        lin.fit(X=depth_vals, y=z_vals)
        if lin.coef_[0] < 0:
            depth = depth * np.mean(
                [z_ / depth_ for z_, depth_ in zip(z_vals, depth_vals)])
        else:
            depth = depth * lin.coef_[0] + lin.intercept_

    if save_dir:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_subdir = img_path.split('/')[-1].split('.')[0]
        os.mkdir(os.path.join(save_dir, save_subdir))
        cv2.imwrite(os.path.join(save_dir, save_subdir, 'real_img.png'), img)
        np.save(os.path.join(save_dir, save_subdir, 'depth_map.npy'), depth)

    return depth


if __name__ == '__main__':
    params = {
        'encoder': 'vitl',
        'features': 256,
        'out_channels': [256, 512, 1024, 1024]
    }
    model = DepthAnythingV2(**params, max_depth=20.0)
    model.load_state_dict(torch.load(
        'metric_depth/depth_anything_v2_metric_hypersim_vitl.pth'
    ))
    model.to(DEVICE)
    get_depth_map(
        model,
        'D:/pythonProject/seg_and_track/R-D-AC_robotic_integration-feat-seg_and_track/services/seg_and_track/tests/data/images/wp0.png'
    )