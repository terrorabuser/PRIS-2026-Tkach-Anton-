from ultralytics import YOLO
import cv2
import numpy as np
import time
from datetime import datetime

import sys
import os
# Add parent directory to path for sort module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.util import get_car, read_license_plate, write_csv
from sort.sort import *

results = {}

mot_tracker = Sort()

# Dictionary to store last detected license plates for each car (for persistence)
# Format: {car_id: {'text': text, 'crop': crop, 'frame_nmr': frame_nmr, 'bbox': bbox}}
car_license_history = {}
MAX_FRAMES_TO_KEEP = 30  # Keep license plate info for 30 frames (~1 second at 30fps)

# Stop line and traffic light variables
stop_line_points = []  # Will store two points to define the stop line
stop_line_defined = False
traffic_light_state = 'red'  # 'red', 'yellow', 'green'
traffic_light_position = None  # (x, y) position for traffic light display
traffic_light_timer = 0
traffic_light_cycle_time = {'red': 5.0, 'yellow': 2.0, 'green': 5.0}  # seconds for each state
traffic_light_start_time = time.time()

# Vehicle crossing tracking
vehicle_crossed_stop_line = {}  # {car_id: {'crossed': bool, 'crossed_at_frame': int}}
violations_saved = set()  # Track which violations have been saved to avoid duplicates

# Create directory for violation photos
violations_dir = './violations'
os.makedirs(violations_dir, exist_ok=True)

# Mouse callback function for marking stop line
def mouse_callback(event, x, y, flags, param):
    global stop_line_points, stop_line_defined, traffic_light_position
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(stop_line_points) < 2:
            stop_line_points.append((x, y))
            print(f"Stop line point {len(stop_line_points)}: ({x}, {y})")
            
            if len(stop_line_points) == 2:
                stop_line_defined = True
                # Set traffic light position near the stop line (to the right)
                line_mid_x = (stop_line_points[0][0] + stop_line_points[1][0]) // 2
                line_mid_y = (stop_line_points[0][1] + stop_line_points[1][1]) // 2
                traffic_light_position = (line_mid_x + 50, line_mid_y - 100)
                print("Stop line defined! Traffic light will appear after marking.")
                print("Press 'r' for red, 'y' for yellow, 'g' for green, 'a' for auto cycle")

def draw_traffic_light(frame, position, state):
    """Draw an artificial traffic light on the frame"""
    if position is None:
        return
    
    x, y = position
    light_size = 40
    spacing = 50
    
    # Draw traffic light box
    box_width = 80
    box_height = 200
    cv2.rectangle(frame, (x - 10, y - 10), (x + box_width - 10, y + box_height - 10), (50, 50, 50), -1)
    cv2.rectangle(frame, (x - 10, y - 10), (x + box_width - 10, y + box_height - 10), (255, 255, 255), 3)
    
    # Draw red light
    red_color = (0, 0, 255) if state == 'red' else (0, 0, 100)
    cv2.circle(frame, (x + 30, y + 30), light_size // 2, red_color, -1)
    cv2.circle(frame, (x + 30, y + 30), light_size // 2, (255, 255, 255), 2)
    
    # Draw yellow light
    yellow_color = (0, 255, 255) if state == 'yellow' else (0, 150, 150)
    cv2.circle(frame, (x + 30, y + 80), light_size // 2, yellow_color, -1)
    cv2.circle(frame, (x + 30, y + 80), light_size // 2, (255, 255, 255), 2)
    
    # Draw green light
    green_color = (0, 255, 0) if state == 'green' else (0, 100, 0)
    cv2.circle(frame, (x + 30, y + 130), light_size // 2, green_color, -1)
    cv2.circle(frame, (x + 30, y + 130), light_size // 2, (255, 255, 255), 2)
    
    # Draw state text
    cv2.putText(frame, state.upper(), (x - 5, y + box_height + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def point_to_line_distance(point, line_start, line_end):
    """Calculate distance from a point to a line segment"""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Vector from line_start to line_end
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        # Line is a point
        return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    
    # Parameter t for closest point on line
    t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)))
    
    # Closest point on line
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance from point to closest point on line
    return np.sqrt((x0 - closest_x)**2 + (y0 - closest_y)**2)

def check_crossing_stop_line(car_bbox, stop_line_points, previous_positions):
    """Check if a vehicle has crossed the stop line"""
    if len(stop_line_points) != 2:
        return False
    
    x1, y1, x2, y2 = car_bbox
    car_center_x = (x1 + x2) / 2
    car_center_y = (y1 + y2) / 2
    car_bottom_y = max(y1, y2)  # Bottom of the car (closest to stop line)
    
    # Check if car bottom is near the stop line
    line_start = stop_line_points[0]
    line_end = stop_line_points[1]
    
    # Calculate distance from car bottom center to stop line
    distance = point_to_line_distance((car_center_x, car_bottom_y), line_start, line_end)
    
    # Consider crossed if distance is less than threshold (e.g., 30 pixels)
    threshold = 30
    is_crossing = distance < threshold
    
    # Check if car was on one side and now on the other (more robust crossing detection)
    if len(previous_positions) >= 2:
        prev_center = previous_positions[-2]
        curr_center = (car_center_x, car_bottom_y)
        
        # Calculate which side of the line the car is on
        def point_side_of_line(point, line_start, line_end):
            px, py = point
            x1, y1 = line_start
            x2, y2 = line_end
            return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        
        prev_side = point_side_of_line(prev_center, line_start, line_end)
        curr_side = point_side_of_line(curr_center, line_start, line_end)
        
        # If sides changed, car crossed the line
        if prev_side * curr_side < 0 and is_crossing:
            return True
    
    return is_crossing

def save_violation_frame(frame, car_id, frame_nmr, license_text=None):
    """Save a violation frame as a photo"""
    global violations_saved
    
    violation_key = f"{car_id}_{frame_nmr}"
    if violation_key in violations_saved:
        return  # Already saved
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{violations_dir}/violation_car{car_id}_frame{frame_nmr}_{timestamp}.jpg"
    
    # Create annotated frame
    violation_frame = frame.copy()
    
    # Add violation text
    violation_text = f"RED LIGHT VIOLATION - Car {car_id}"
    if license_text:
        violation_text += f" - {license_text}"
    
    # Draw text background
    (text_width, text_height), baseline = cv2.getTextSize(
        violation_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    
    cv2.rectangle(violation_frame, (10, 10), 
                 (20 + text_width, 30 + text_height + baseline), 
                 (0, 0, 0), -1)
    
    # Draw violation text
    cv2.putText(violation_frame, violation_text, (15, 30 + text_height), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    # Draw frame number and timestamp
    frame_info = f"Frame: {frame_nmr} | Time: {timestamp}"
    cv2.putText(violation_frame, frame_info, (15, 80 + text_height), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    cv2.imwrite(filename, violation_frame)
    violations_saved.add(violation_key)
    print(f"Violation saved: {filename}")

# load models
coco_model = YOLO('yolov8n.pt')
# Note: license_plate_detector.pt needs to be placed in models/ folder
# You can download it from the repository's Patreon or train your own
try:
    license_plate_detector = YOLO('./models/license_plate_detector.pt')
except:
    print("Warning: license_plate_detector.pt not found. Please add it to models/ folder.")
    license_plate_detector = None

# load video
cap = cv2.VideoCapture('./data/raw/video_2.mp4')

# Get video properties
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video resolution: {video_width}x{video_height}, FPS: {fps}")

# Create window for main video with original size (no scaling)
cv2.namedWindow('License Plate Recognition', cv2.WINDOW_NORMAL)
cv2.resizeWindow('License Plate Recognition', video_width, video_height)
cv2.setMouseCallback('License Plate Recognition', mouse_callback)

# Instructions
print("\n" + "="*60)
print("INSTRUCTIONS:")
print("1. Click TWO points on the video to mark the stop line")
print("2. After marking, traffic light will appear")
print("3. Press 'r' for red, 'y' for yellow, 'g' for green")
print("4. Press 'a' for automatic traffic light cycling")
print("5. Press 'c' to clear stop line and start over")
print("6. Press 'q' to quit")
print("="*60 + "\n")

# Vehicle position history for crossing detection
vehicle_positions = {}  # {car_id: [(x, y), ...]}

vehicles = [2, 3, 5, 7]

# Auto cycle traffic light flag
auto_cycle = False

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        # Update traffic light state (auto cycle)
        if auto_cycle and stop_line_defined:
            current_time = time.time()
            elapsed = current_time - traffic_light_start_time
            
            if elapsed >= traffic_light_cycle_time[traffic_light_state]:
                # Switch to next state
                if traffic_light_state == 'red':
                    traffic_light_state = 'green'
                elif traffic_light_state == 'green':
                    traffic_light_state = 'yellow'
                else:  # yellow
                    traffic_light_state = 'red'
                traffic_light_start_time = current_time
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and stop_line_defined:
            traffic_light_state = 'red'
            traffic_light_start_time = time.time()
            auto_cycle = False
            print("Traffic light set to RED")
        elif key == ord('y') and stop_line_defined:
            traffic_light_state = 'yellow'
            traffic_light_start_time = time.time()
            auto_cycle = False
            print("Traffic light set to YELLOW")
        elif key == ord('g') and stop_line_defined:
            traffic_light_state = 'green'
            traffic_light_start_time = time.time()
            auto_cycle = False
            print("Traffic light set to GREEN")
        elif key == ord('a') and stop_line_defined:
            auto_cycle = not auto_cycle
            traffic_light_start_time = time.time()
            print(f"Auto cycle: {'ON' if auto_cycle else 'OFF'}")
        elif key == ord('c'):
            stop_line_points = []
            stop_line_defined = False
            traffic_light_position = None
            vehicle_crossed_stop_line = {}
            vehicle_positions = {}
            print("Stop line cleared")
        
        # Draw stop line if defined
        if len(stop_line_points) == 2:
            cv2.line(frame, stop_line_points[0], stop_line_points[1], (0, 0, 255), 5)
            # Draw points
            cv2.circle(frame, stop_line_points[0], 8, (255, 0, 0), -1)
            cv2.circle(frame, stop_line_points[1], 8, (255, 0, 0), -1)
            cv2.putText(frame, "STOP LINE", 
                       (stop_line_points[0][0] - 50, stop_line_points[0][1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Draw traffic light if stop line is defined
        if stop_line_defined and traffic_light_position:
            draw_traffic_light(frame, traffic_light_position, traffic_light_state)
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Dictionary to store license plate info for each car
        car_license_info = {}
        
        # detect license plates first
        if license_plate_detector is not None:
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    # Add larger padding to capture full license plate (30% on each side for better reading)
                    frame_h, frame_w = frame.shape[:2]
                    padding_x = int((x2 - x1) * 0.3)  # 30% padding horizontally
                    padding_y = int((y2 - y1) * 0.3)  # 30% padding vertically
                    
                    # Calculate crop coordinates with padding, ensuring they stay within frame bounds
                    crop_x1 = max(0, int(x1) - padding_x)
                    crop_y1 = max(0, int(y1) - padding_y)
                    crop_x2 = min(frame_w, int(x2) + padding_x)
                    crop_y2 = min(frame_h, int(y2) + padding_y)
                    
                    # Ensure valid coordinates
                    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                        # crop license plate with padding
                        license_plate_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2, :].copy()
                        
                        # Check if crop is valid
                        if license_plate_crop.size > 0:
                            # process license plate
                            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                            # read license plate number
                            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                            # Draw license plate bounding box (red, thicker)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            
                            # Store license plate info for this car
                            if license_plate_text is not None:
                                car_license_info[car_id] = {
                                    'text': license_plate_text,
                                    'crop': license_plate_crop,
                                    'bbox': [x1, y1, x2, y2],
                                    'score': score,
                                    'text_score': license_plate_text_score
                                }
                                
                                # Also save to history for persistence across frames
                                car_license_history[car_id] = {
                                    'text': license_plate_text,
                                    'crop': license_plate_crop,
                                    'frame_nmr': frame_nmr,
                                    'bbox': [x1, y1, x2, y2]
                                }
                        
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}
                else:
                    # Draw license plate even if not assigned to car (yellow)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, 'Unassigned', (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw vehicle bounding boxes with license plate info
        for track in track_ids:
            xcar1, ycar1, xcar2, ycar2, car_id = track
            xcar1, ycar1, xcar2, ycar2 = int(xcar1), int(ycar1), int(xcar2), int(ycar2)
            car_id = int(car_id)
            
            # Track vehicle position for crossing detection
            if car_id not in vehicle_positions:
                vehicle_positions[car_id] = []
            
            car_center_x = (xcar1 + xcar2) / 2
            car_bottom_y = max(ycar1, ycar2)
            vehicle_positions[car_id].append((car_center_x, car_bottom_y))
            
            # Keep only last 5 positions
            if len(vehicle_positions[car_id]) > 5:
                vehicle_positions[car_id].pop(0)
            
            # Initialize crossing tracking for new vehicles
            if car_id not in vehicle_crossed_stop_line:
                vehicle_crossed_stop_line[car_id] = {'crossed': False, 'crossed_at_frame': -1}
            
            # Check if vehicle crosses stop line
            violation_detected = False
            if stop_line_defined and len(stop_line_points) == 2:
                is_crossing = check_crossing_stop_line(
                    [xcar1, ycar1, xcar2, ycar2], 
                    stop_line_points,
                    vehicle_positions[car_id]
                )
                
                if is_crossing and not vehicle_crossed_stop_line[car_id]['crossed']:
                    vehicle_crossed_stop_line[car_id]['crossed'] = True
                    vehicle_crossed_stop_line[car_id]['crossed_at_frame'] = frame_nmr
                    
                    # Check if traffic light is red
                    if traffic_light_state == 'red':
                        violation_detected = True
                        # Get license plate text if available
                        license_text = None
                        if car_id in car_license_info:
                            license_text = car_license_info[car_id]['text']
                        elif car_id in car_license_history:
                            license_text = car_license_history[car_id]['text']
                        
                        # Save violation frame
                        save_violation_frame(frame, car_id, frame_nmr, license_text)
                        print(f"VIOLATION DETECTED! Car {car_id} crossed stop line on RED light (Frame {frame_nmr})")
            
            # Draw vehicle bounding box (red if violation, green otherwise)
            box_color = (0, 0, 255) if violation_detected else (0, 255, 0)
            box_thickness = 6 if violation_detected else 4
            cv2.rectangle(frame, (xcar1, ycar1), (xcar2, ycar2), box_color, box_thickness)
            
            # Draw violation warning if detected
            if violation_detected:
                cv2.putText(frame, 'VIOLATION!', (xcar1, ycar1 - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            
            # Check if this car has license plate info
            if car_id in car_license_info:
                license_info = car_license_info[car_id]
                license_text = license_info['text']
                license_crop = license_info['crop']
                
                # Calculate position for license plate crop above the car (large format)
                crop_h, crop_w = license_crop.shape[:2]
                
                # Scale up significantly for better readability (3-4x larger)
                car_width = xcar2 - xcar1
                target_width = max(300, min(500, car_width * 2.5))  # 2.5x car width, min 300px, max 500px
                scale_factor = target_width / crop_w if crop_w > 0 else 1.0
                
                # Ensure minimum scale for readability
                if scale_factor < 2.0:
                    scale_factor = 2.0
                
                new_crop_w = int(crop_w * scale_factor)
                new_crop_h = int(crop_h * scale_factor)
                
                # Ensure minimum size for readability
                if new_crop_w < 300:
                    new_crop_w = 300
                if new_crop_h < 150:
                    new_crop_h = 150
                
                # Resize the crop with high-quality interpolation
                license_crop_resized = cv2.resize(license_crop, (new_crop_w, new_crop_h), 
                                                 interpolation=cv2.INTER_CUBIC)
                
                # Enhance contrast for better readability
                lab = cv2.cvtColor(license_crop_resized, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                l = clahe.apply(l)
                license_crop_resized = cv2.merge([l, a, b])
                license_crop_resized = cv2.cvtColor(license_crop_resized, cv2.COLOR_LAB2BGR)
                
                # Position above the car (centered)
                crop_x = xcar1 + (xcar2 - xcar1 - new_crop_w) // 2
                crop_y = ycar1 - new_crop_h - 60  # 60 pixels above car for text
                
                # Make sure crop doesn't go outside frame
                if crop_y < 0:
                    crop_y = ycar2 + 10  # Put below car if no space above
                
                # Ensure crop stays within frame bounds
                if crop_x < 0:
                    crop_x = 5
                if crop_x + new_crop_w > frame.shape[1]:
                    crop_x = frame.shape[1] - new_crop_w - 5
                
                # Draw white background for crop
                cv2.rectangle(frame, (crop_x - 3, crop_y - 3), 
                            (crop_x + new_crop_w + 3, crop_y + new_crop_h + 3), 
                            (255, 255, 255), -1)
                
                # Place the license plate crop
                if (crop_y + new_crop_h < frame.shape[0] and 
                    crop_x + new_crop_w < frame.shape[1] and
                    crop_y >= 0 and crop_x >= 0):
                    frame[crop_y:crop_y + new_crop_h, crop_x:crop_x + new_crop_w] = license_crop_resized
                
                # Draw border around crop
                cv2.rectangle(frame, (crop_x - 3, crop_y - 3), 
                            (crop_x + new_crop_w + 3, crop_y + new_crop_h + 3), 
                            (255, 0, 0), 3)
                
                # Draw license plate text above the crop (large format)
                text_y = crop_y - 15
                if text_y < 25:
                    text_y = crop_y + new_crop_h + 35  # Put below crop if no space above
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    license_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                
                # Draw text background
                text_bg_x1 = crop_x - 3
                text_bg_x2 = crop_x + max(text_width, new_crop_w) + 3
                text_bg_y1 = text_y - text_height - 8
                text_bg_y2 = text_y + 8
                
                cv2.rectangle(frame, 
                            (text_bg_x1, text_bg_y1), 
                            (text_bg_x2, text_bg_y2), 
                            (0, 0, 0), -1)
                
                # Draw license plate text (large and bold)
                cv2.putText(frame, license_text, (crop_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                # Check if we have license plate info from previous frames
                if car_id in car_license_history:
                    frames_since_detection = frame_nmr - car_license_history[car_id]['frame_nmr']
                    if frames_since_detection <= MAX_FRAMES_TO_KEEP:
                        # Use cached license plate info
                        license_info = car_license_history[car_id]
                        license_text = license_info['text']
                        license_crop = license_info['crop']
                        
                        # Calculate position for license plate crop above the car (large format)
                        crop_h, crop_w = license_crop.shape[:2]
                        
                        # Scale up significantly for better readability
                        car_width = xcar2 - xcar1
                        target_width = max(300, min(500, car_width * 2.5))
                        scale_factor = target_width / crop_w if crop_w > 0 else 1.0
                        
                        if scale_factor < 2.0:
                            scale_factor = 2.0
                        
                        new_crop_w = int(crop_w * scale_factor)
                        new_crop_h = int(crop_h * scale_factor)
                        
                        if new_crop_w < 300:
                            new_crop_w = 300
                        if new_crop_h < 150:
                            new_crop_h = 150
                        
                        # Resize the crop with high-quality interpolation
                        license_crop_resized = cv2.resize(license_crop, (new_crop_w, new_crop_h), 
                                                         interpolation=cv2.INTER_CUBIC)
                        
                        # Enhance contrast for better readability
                        lab = cv2.cvtColor(license_crop_resized, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                        l = clahe.apply(l)
                        license_crop_resized = cv2.merge([l, a, b])
                        license_crop_resized = cv2.cvtColor(license_crop_resized, cv2.COLOR_LAB2BGR)
                        
                        # Position above the car (centered)
                        crop_x = xcar1 + (xcar2 - xcar1 - new_crop_w) // 2
                        crop_y = ycar1 - new_crop_h - 60
                        
                        if crop_y < 0:
                            crop_y = ycar2 + 10
                        
                        if crop_x < 0:
                            crop_x = 5
                        if crop_x + new_crop_w > frame.shape[1]:
                            crop_x = frame.shape[1] - new_crop_w - 5
                        
                        # Draw white background for crop
                        cv2.rectangle(frame, (crop_x - 3, crop_y - 3), 
                                    (crop_x + new_crop_w + 3, crop_y + new_crop_h + 3), 
                                    (255, 255, 255), -1)
                        
                        # Place the license plate crop
                        if (crop_y + new_crop_h < frame.shape[0] and 
                            crop_x + new_crop_w < frame.shape[1] and
                            crop_y >= 0 and crop_x >= 0):
                            frame[crop_y:crop_y + new_crop_h, crop_x:crop_x + new_crop_w] = license_crop_resized
                        
                        # Draw border around crop
                        cv2.rectangle(frame, (crop_x - 3, crop_y - 3), 
                                    (crop_x + new_crop_w + 3, crop_y + new_crop_h + 3), 
                                    (255, 0, 0), 3)
                        
                        # Draw license plate text above the crop (large format)
                        text_y = crop_y - 15
                        if text_y < 25:
                            text_y = crop_y + new_crop_h + 35
                        
                        # Get text size for background
                        (text_width, text_height), baseline = cv2.getTextSize(
                            license_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                        
                        # Draw text background
                        text_bg_x1 = crop_x - 3
                        text_bg_x2 = crop_x + max(text_width, new_crop_w) + 3
                        text_bg_y1 = text_y - text_height - 8
                        text_bg_y2 = text_y + 8
                        
                        cv2.rectangle(frame, 
                                    (text_bg_x1, text_bg_y1), 
                                    (text_bg_x2, text_bg_y2), 
                                    (0, 0, 0), -1)
                        
                        # Draw license plate text (large and bold)
                        cv2.putText(frame, license_text, (crop_x, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    else:
                        # Remove old license plate info
                        del car_license_history[car_id]
                        # Draw car ID if no license plate detected
                        cv2.putText(frame, f'Car {car_id}', (xcar1, ycar1 - 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                else:
                    # Draw car ID if no license plate detected
                    cv2.putText(frame, f'Car {car_id}', (xcar1, ycar1 - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Display frame info (larger text)
        info_y = 40
        cv2.putText(frame, f'Frame: {frame_nmr}', (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        info_y += 40
        cv2.putText(frame, f'Vehicles: {len(track_ids)}', (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        # Display stop line status
        info_y += 40
        if stop_line_defined:
            cv2.putText(frame, 'Stop Line: DEFINED', (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'Stop Line: Click 2 points to define', (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Display traffic light status
        if stop_line_defined:
            info_y += 40
            light_status = f'Traffic Light: {traffic_light_state.upper()}'
            if auto_cycle:
                light_status += ' (AUTO)'
            light_color = (0, 0, 255) if traffic_light_state == 'red' else \
                         (0, 255, 255) if traffic_light_state == 'yellow' else (0, 255, 0)
            cv2.putText(frame, light_status, (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, light_color, 3)
        
        # Display violation count
        info_y += 40
        violation_count = len(violations_saved)
        cv2.putText(frame, f'Violations: {violation_count}', (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Display the frame in original resolution (no scaling)
        cv2.imshow('License Plate Recognition', frame)
    else:
        # If frame reading failed, break the loop
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# write results
write_csv(results, './test.csv')
print(f"\n" + "="*60)
print("Processing complete!")
print("="*60)
print(f"Total frames processed: {frame_nmr + 1}")
print(f"Total vehicles tracked: {len(set([car_id for frame_data in results.values() for car_id in frame_data.keys()]))}")
print(f"Total violations detected: {len(violations_saved)}")
print(f"Violation photos saved in: {violations_dir}/")
print("="*60)