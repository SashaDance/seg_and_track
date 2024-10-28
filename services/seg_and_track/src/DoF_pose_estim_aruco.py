import numpy as np
import cv2

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''

    # https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
    # usage example https://github.com/Menginventor/aruco_example_cv_4.8.0

    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash



import cv2
import numpy as np

def draw_pose_axis(image, rvec, tvec, camera_matrix, dist_coeffs):
    # Ensure tvec is a 1x3 float32 array
    tvec = np.array(tvec, dtype=np.float32).reshape(1, 3)  # Reshape to 1x3 if necessary
    
    # Define the axis points in 3D space
    axis_points = np.array([[0, 0, 0],   # Origin
                            [1, 0, 0],   # X-axis
                            [0, 1, 0],   # Y-axis
                            [0, 0, 1]],  # Z-axis
                           dtype=np.float32)

    # Project the axis points to 2D
    projected_axis_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

    # Draw the axes
    origin = tuple(projected_axis_points[0].ravel())  # Origin in 2D
    axis_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Colors for X, Y, Z axes

    for i in range(1, 4):  # Start from 1 to avoid the origin point
        end_point = tuple(projected_axis_points[i].ravel())  # Convert to tuple
        cv2.line(image, origin, end_point, axis_colors[i - 1], 3)  # Draw line from origin to end point

    return image






