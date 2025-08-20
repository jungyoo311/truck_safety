'''
Lucas-Kanade optical flow tracker for vehicle motion analysis calculating speed and acceleration from feature point movement.
Implements corner detection using Shi-Tomasi algorithm and tracks feature points across frames for motion estimation.
Real-time tracking with speed/acceleration computation for forward collision warning and vehicle behavior analysis.
'''
import numpy as np
import time
import cv2
class LucasKanadeTracker:
    def __init__(self):
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.prev_gray = None
        self.prev_pts = None
        self.prev_time = None
        self.prev_speed = None
        self.findNew = True

    def initialize(self, frame, points=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #shitomashi corner detections. R = min(lambda1, lambda2)
        # 
        if points is not None:
            self.prev_pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        else:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        self.prev_gray = gray
        self.prev_time = time.time()
        self.prev_speed = 0
        self.findNew = False

    def track(self, frame):
            """Track points using Lucas-Kanade Optical Flow."""
            if self.prev_pts is None or self.prev_gray is None:
                print("[DEBUG] Tracker has no previous frame.")
                return frame, 0, 0  # No movement detected

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Lucas-Kanade optical flow
            new_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **self.lk_params)

            if new_pts is None or status is None:
                print("[ERROR] Optical flow failed.")
                return frame, 0, 0

            # Compute speed & acceleration
            current_time = time.time()
            dt = current_time - self.prev_time if self.prev_time else 1.0
            distances = np.linalg.norm(new_pts - self.prev_pts, axis=2).flatten()
            avg_movement = np.mean(distances) if len(distances) > 0 else 0.0

            # compute speed in px/sec
            speed = avg_movement / dt

            # compute acceleration
            if self.prev_speed is not None:
                acceleration = (speed - self.prev_speed) / dt
            else:
                acceleration = 0.0

            # update times and speeds
            self.prev_time = current_time
            self.prev_speed = speed

            # Print debug info
            print(f"INSIDE of LUCAS, prev_pts is: {self.prev_pts}")
            print(f"_*_*new_pts: {new_pts}, status: {status}, err:{err}")
            print(f"[DEBUG] Speed: {speed:.2f} px/s, Acceleration: {acceleration:.2f} px/s²")

            # Update tracker state
            self.prev_gray = gray
            self.prev_pts = new_pts[status == 1].reshape(-1, 1, 2)  # Keep only good points

            return frame, speed, acceleration
