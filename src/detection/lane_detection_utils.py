'''
Forward collision warning system combining multiple lane detection methods: edge detection, line detection, and AI segmentation.
Tracks vehicle speed and acceleration to calculate collision risk and trigger safety alerts.
Real-time video processing pipeline for automotive safety applications.
'''
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo, sys, time, queue,threading
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# Adjust path to the new location of hailo-apps-infra
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hailo-apps-infra")))

from truck_safety.src.utils.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from truck_safety.src.detection.detection_pipeline import GStreamerDetectionApp
from truck_safety.src.tracking.lucas_kanade_tracker import LucasKanadeTracker
from utils import HailoAsyncInference
tracker = LucasKanadeTracker()
DANGER_ACCELERATION_THRESHOLD = -5

#queue for feeding lane detection inference
lane_input_q = queue.Queue(maxsize=3)
lane_output_q = queue.Queue(maxsize=3)

#load HailoAsyncInference for lane detction
HEF_PATH = "/home/jung/Desktop/hailo-rpi5-examples/truck-safety/resources/yolov10s.hef"
lane_inference = HailoAsyncInference(hef_path='/home/jung/Desktop/hailo-rpi5-examples/truck-safety/resources/yolov10s.hef', input_queue=lane_input_q, output_queue=lane_output_q,batch_size=1)
#Run inference in a separate thread
def start_lane_inference():
    lane_inference.run()
threading.Thread(target=start_lane_inference, daemon=True).start()

from truck_safety.src.detection.lane_detection_utils import UFLDProcessing

# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_skip = 2 #process every 2nd frame
        #plotting values
        self.speeds = []
        self.times = []
        self.start_time = time.time()
        self.log_filename = "speed_log.csv"

    # making log file in csv format
    def visualize(self, speed):
        current_time = time.time() - self.start_time
        self.speeds.append(speed)
        self.times.append(current_time)

        # Log speed to CSV
        with open(self.log_filename, "a") as f:
            f.write(f"{current_time}, {speed}\n")

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

'''
 * To add this filter to your pipeline, you would typically include the following in your code:
 * remover_so_path = os.path.join(self.current_path, '../resources/libremove_labels.so')
 * # adjust the path to the labels_to_remove.txt file
 * labels_to_remove_path = os.path.join(self.current_path, '../resources/labels_to_remove.txt')
 * 'hailofilter name=remover so-path=remover_so_path config-path=labels_to_remove_path qos=false ! '
'''

def app_callback_custom(pad, info, user_data):
    
    user_data.use_frame = True
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format, width, height = get_caps_from_pad(pad)
    # print(f"_*_*format: {format}, width: {width}, height: {height}")
    frame = get_numpy_from_buffer(buffer, format, width, height)
    # if frame is not None:
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    print(f"[DEBUG] Total Detections: {len(detections)}")

    # Find the frontmost vehicle
    min_y = float('inf')
    front_vehicle_centroid = None
    detection_count = 0
    front_vehicle_detection = None
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()  # Fetch bbox object
        confidence = detection.get_confidence()
        # print(f"label: {label}, bbox: {bbox.xmin}, confidence: {confidence}")

        if bbox is not None and label in ["car", "truck"] and confidence >= 0.6:
            x1, y1 = int(bbox.xmin() * width), int(bbox.ymin() * height)
            x2, y2 = int(bbox.xmax() * width), int(bbox.ymax() * height)
            # print(f"[DEBUG] Detection: Label={label}, BBox=({x1}, {y1}, {x2}, {y2}), Confidence={confidence:.2f}")
            centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2    
            # print(f"[DEBUG] Selected Vehicle Centroid: ({centroid_x}, {centroid_y})")

            if y1 < min_y:  # Closer to top means further ahead
                min_y = y1
                front_vehicle_centroid = (centroid_x, centroid_y)
                front_vehicle_detection = detection

    if tracker.prev_pts is None or len(tracker.prev_pts) == 0:
        print("[DEBUG] Tracker was empty, initializing with front vehicle.")
        tracker.initialize(frame, [front_vehicle_centroid])
    else:
        # Track motion and compute speed/acceleration
        frame, speed, acceleration = tracker.track(frame)
        h, w, _ = frame.shape
        x_text_ = w - 300  
        y_speed_ = 40      
        y_accel_ = 80
        y_warning_ = 120
        print(f"[DEBUG] Speed: {speed:.2f} px/s, Acceleration: {acceleration:.2f} px/s^2")
        cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # Red dot at centroid
        cv2.putText(
            frame,
            f"Speed={speed:.2f} px/s",
            (x_text_, y_speed_),           # <- origin (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 255), 2                             
        )
        cv2.putText(
            frame,
            f"Acc={acceleration:.2f} px/s^2",
            (x_text_, y_accel_),           
            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 255), 2
        )
        #user_data.visualize(speed) #log speed
        if speed < 0.1:
            print("[WARNING] Tracking is stuck! The object might not be moving or tracking is failing.")
            cv2.putText(
            frame,
            f"BRAKE!!!",
            (x_text_, y_warning_),           
            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 255), 2
        )
            
    if user_data.use_frame:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        user_data.set_frame(frame)
    # print(f"[DEBUG] Frame Count: {user_data.get_count()}")
    # print(f"[DEBUG] Detected Objects: {len(detections)}")

    return Gst.PadProbeReturn.OK

############################################################################
# Lane Detection Helpers
############################################################################
def detect_lane_mask(frame_bgr):
    """
    Very basic lane detection mask: 
      1) Focus on bottom half of the image 
      2) Use Canny edges + Hough lines 
      3) Convert lines to a binary mask
    """
    h, w = frame_bgr.shape[:2]

    # Convert to gray, build an ROI focusing on bottom half
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    roi_vertices = np.array([[
        (0, h),
        (0, int(h * 0.5)),
        (w, int(h * 0.5)),
        (w, h)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    roi = cv2.bitwise_and(gray, mask)

    # Edges + Hough lines
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=40, maxLineGap=100)

    # Draw lines on a blank canvas
    line_img = np.zeros_like(frame_bgr)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Convert to a lane_mask
    line_gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    _, lane_mask = cv2.threshold(line_gray, 1, 255, cv2.THRESH_BINARY)
    return lane_mask

def get_lane_polygon(lane_mask, vehicle_postision = None, danger_zone_ratio=0.3):
    """
    Convert lane_mask into a polygon. 
    For simplicity, we pick the largest contour.

    lane_mask: binary mask of lane markings
    vehicle_position: (x, y) tuple of the vehicle's position in the image
    danger_zone_ratio: proportion of the image height to consider as the "danger zone"

    Returns:
        dange_zone_polygon: approximated polygon for the danger area.

    """
    contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    #define the danger zone regions
    h, w = lane_mask.shape
    danger_zone_y = int(h * danger_zone_ratio)
    danger_contours = [cnt for cnt in contours if any(pt[0][1] >= danger_zone_y for pt in cnt)]

    if not danger_contours:
        return None
    largest_contour_area = max(danger_contours, key=cv2.contourArea)
    danger_zone_polygon = cv2.approxPolyDP(largest_contour_area, 4, True) #approximate the polygon
    return danger_zone_polygon

def point_in_polygon(cx, cy, polygon):
    """
    Returns True if (cx, cy) is inside polygon using cv2.pointPolygonTest.
    """
    dist = cv2.pointPolygonTest(polygon, (cx, cy), False)
    # dist >= 0 => inside or on the boundary
    return (dist >= 0)
def define_danger_zone(frame):
    h, w = frame.shape[:2]
    polygon = np.array([
        [int(0.45 * w), int(h * 0.55)],
        [int(0.55* w), int(h * 0.55)],
        [int(0.8 * w), h],
        [int(0.2 * w), h],
    ], np.int32)
    return polygon
def is_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0
def detect_lane_masks(roi, width, height):
    """
    Extract lane segmentation masks from Hailo's inference.

    Args:
        roi: Hailo Region of Interest (ROI) containing detected objects.
        width (int): Frame width.
        height (int): Frame height.

    Returns:
        lane_mask (np.array): A binary mask of lane areas.
    """
    lane_mask = np.zeros((height, width), dtype=np.uint8)

    # Get instance segmentation results
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        # Assuming lanes are labeled as "lane" or similar in the model
        if label in ["lane", "road_marking", "lane_boundary"] and confidence >= 0.6:
            masks = detection.get_objects_typed(hailo.HAILO_CONF_CLASS_MASK)
            if len(masks) != 0:
                mask = masks[0]

                # Reshape the mask into its original dimensions
                mask_height, mask_width = mask.get_height(), mask.get_width()
                data = np.array(mask.get_data()).reshape((mask_height, mask_width))

                # Resize mask to the bounding box size
                roi_width = int(bbox.width() * width)
                roi_height = int(bbox.height() * height)
                resized_mask = cv2.resize(data, (roi_width, roi_height), interpolation=cv2.INTER_LINEAR)

                # Calculate the ROI coordinates
                x_min, y_min = int(bbox.xmin() * width), int(bbox.ymin() * height)
                x_max, y_max = x_min + roi_width, y_min + roi_height

                # Ensure dimensions are within valid range
                y_min, x_min = max(y_min, 0), max(x_min, 0)
                y_max, x_max = min(y_max, height), min(x_max, width)

                # Apply mask overlay
                if x_max > x_min and y_max > y_min:
                    lane_mask[y_min:y_max, x_min:x_max] = (resized_mask[:y_max - y_min, :x_max - x_min] > 0.5) * 255

    return lane_mask

# Predefined colors (BGR format)
LIGHT_GREEN = (144, 238, 144)  # Light Green for lane regions
DARK_GREEN = (0, 100, 0)  # Dark Green for detected lane edges
# already handles reading from the camera or video source. applying the Hailo pipeline and calling thsi for each frame.
def app_callback_fcw(pad, info, user_data):
    """
    1. Convert GStreamer buffer to Numpy RGB
    2. Run lane detection using UFLD
    3. Draw lane lines on a debug overlay
    4. Extract Hailo detection results from the buffer
    5. Identify front vehicle within the detected lane
    6. Track the front vehilce and annotate
    7. Show final debug frame in user_data.set_frame()

    I want to draw danger zone with lane detections. I want danger zone in front of the car.
    1. prototype current version: static danger zone; current fcw.py code
    2. moving danger zone based on the speed of current vehicle and front vehicle speed and 
    distance based on lucas-kanade method.

    one red dot in the iou zone? will be focused, in the case of lane change from other lane, 
    up to 4 interested cars will be exist. the clostest car's red dot is extremly closer 
    to the bottom center of it's bounding box, it alerts FCW and slow!c

    1. should i implement 1-a: danger zone red dot tracking method 
                          1-b: lane detection method -> to filter out cars in adjacent lanes
                          question: I have current speed of the ego vehicle how can i use that to enhance fcw? now too much false positives and false negatives
                          Testing: compare results for 2-a and 2-b
                          2-a: TTC approach with current vehicle speed - lucas-kanade 
                          2-b: TTC approach with current vehicle speed - relative speed from the log file
    """
    buffer = info.get_buffer()
    if not buffer:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format, width, height = get_caps_from_pad(pad)
    frame_rgb = get_numpy_from_buffer(buffer, format, width, height)
    if frame_rgb is None:
        return Gst.PadProbeReturn.OK

    # Convert to BGR for normal OpenCV usage
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    debug_frame = frame_bgr.copy()

    # lane_points = ufld_processing.get_coordinates(frame_resized) # Run inference
    lane_points = None
    if lane_points:
        for lane in lane_points:
            points = np.array(lane, np.int32)
            cv2.polylines(debug_frame, [points], False, (0, 255, 0), 2)
    # Define and Draw Danger zones
    danger_zone_polygon = define_danger_zone(debug_frame)
    cv2.polylines(debug_frame, [danger_zone_polygon], True, (255,0,0), 2)
    # Vehicle detection 
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    for det in detections:
        if det.get_label() in ["car", "truck"] and det.get_confidence() >= 0.6:
            bbox = det.get_bbox()
            centroid_x = int((bbox.xmin() + bbox.xmax()) / 2 * debug_frame.shape[1])
            centroid_y = int((bbox.ymin() + bbox.ymax()) / 2 * debug_frame.shape[0])

            # Draw centroid
            cv2.circle(debug_frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

            # FCW alert if centroid is in danger zone
            if is_inside_polygon((centroid_x, centroid_y), define_danger_zone(debug_frame)):
                cv2.putText(debug_frame, "FCW ALERT!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
    if user_data.use_frame:
        user_data.set_frame(debug_frame)

    return Gst.PadProbeReturn.OK
def app_callback_fcw2(pad, info, user_data):
    buffer = info.get_buffer()
    if not buffer:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format, width, height = get_caps_from_pad(pad)
    frame_rgb = get_numpy_from_buffer(buffer, format, width, height)
    if frame_rgb is None:
        return Gst.PadProbeReturn.OK

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    debug_frame = frame_bgr.copy()

    lane_mask = np.zeros((height, width), dtype=np.uint8)
    try:
        _, lane_output = lane_output_q.get_nowait()
        lane_mask = lane_output  # Assuming output is a lane segmentation mask
    except queue.Empty:
        pass  # Use the empty lane mask if no result is available

    # Overlay detected lane markings
    lane_overlay = np.zeros_like(frame_bgr)
    lane_overlay[lane_mask > 0] = (144, 238, 144)  # Light Green for lane areas
    edges = cv2.Canny(lane_mask, 50, 150)
    lane_overlay[edges > 0] = (0, 100, 0)  # Dark Green for lane edges
    debug_frame = cv2.addWeighted(debug_frame, 0.7, lane_overlay, 0.3, 0)

    # Detect Vehicles using Hailo
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    min_y = float('inf')
    front_vehicle_centroid = None
    front_vehicle_detection = None

    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        if bbox is not None and label in ["car", "truck"] and confidence >= 0.6:
            x1, y1 = int(bbox.xmin() * width), int(bbox.ymin() * height)
            x2, y2 = int(bbox.xmax() * width), int(bbox.ymax() * height)
            centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Filter out vehicles outside the ego lane
            if lane_mask[centroid_y, centroid_x] == 0:
                continue  # Skip vehicles outside the lane

            # Identify the closest front vehicle
            if y1 < min_y:
                min_y = y1
                front_vehicle_centroid = (centroid_x, centroid_y)
                front_vehicle_detection = detection

    # Draw Bounding Box for the Selected Front Vehicle
    if front_vehicle_detection:
        bbox = front_vehicle_detection.get_bbox()
        x1, y1 = int(bbox.xmin() * width), int(bbox.ymin() * height)
        x2, y2 = int(bbox.xmax() * width), int(bbox.ymax() * height)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(debug_frame, "Front Vehicle", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Ensure front_vehicle_centroid is valid before initializing tracker
    if front_vehicle_centroid is not None:
        tracker.initialize(frame_bgr, [front_vehicle_centroid])
    else:
        print("[WARNING] No valid front vehicle detected for tracking.")

    # Perform Tracking if Tracker is Initialized
    if tracker.prev_pts is not None and len(tracker.prev_pts) > 0:
        frame_bgr, speed, acceleration = tracker.track(frame_bgr)
        cv2.putText(debug_frame, f"Speed: {speed:.2f} px/s", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(debug_frame, f"Acc: {acceleration:.2f} px/s^2", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if acceleration < DANGER_ACCELERATION_THRESHOLD:
            cv2.putText(debug_frame, "FCW ALERT!", (50, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    if user_data.use_frame:
        user_data.set_frame(debug_frame)

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    user_data.use_frame = True
    app = GStreamerDetectionApp(app_callback_fcw, user_data)
    app.run()
