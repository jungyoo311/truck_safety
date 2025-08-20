'''
Camera calibration utility using chessboard pattern detection to determine camera intrinsic parameters and distortion coefficients.
usage: Single image calibration test for dashcam setup verification and camera parameter extraction.
'''
import cv2
import numpy as np

# Prepare object points (3D points in real world)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

# Load your dashcam image
img = cv2.imread("/home/jung/Desktop/hailo-rpi5-examples/truck-safety/resources/chessboard_dashcam_cropped.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow("gray img", gray)
# Find chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
print(f"ret: {ret}, corners: {corners}")
if ret:
    objpoints.append(objp)
    imgpoints.append(corners)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist)

    if np.all(np.abs(dist) < 0.1):
        print("Camera follows the pinhole model!")
    else:
        print("Camera has significant distortion!")
else:
    print("Chessboard not detected. Try a clearer image.")
