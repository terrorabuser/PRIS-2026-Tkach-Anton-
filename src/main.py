from ultralytics import YOLO
import cv2
import numpy as np

import sys
import os
# Add parent directory to path for sort module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.util import get_car, read_license_plate, write_csv
from sort.sort import *

results = {}

mot_tracker = Sort()

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

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
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

        # Draw vehicle bounding boxes
        for track in track_ids:
            xcar1, ycar1, xcar2, ycar2, car_id = track
            xcar1, ycar1, xcar2, ycar2 = int(xcar1), int(ycar1), int(xcar2), int(ycar2)
            car_id = int(car_id)
            
            # Draw vehicle bounding box (green)
            cv2.rectangle(frame, (xcar1, ycar1), (xcar2, ycar2), (0, 255, 0), 2)
            # Draw car ID
            cv2.putText(frame, f'Car {car_id}', (xcar1, ycar1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # detect license plates
        if license_plate_detector is not None:
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:

                    # crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    # Draw license plate bounding box (red)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    if license_plate_text is not None:
                        # Draw license plate text
                        cv2.putText(frame, license_plate_text, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}
                    else:
                        # Draw "Reading..." if text not recognized
                        cv2.putText(frame, 'Reading...', (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                else:
                    # Draw license plate even if not assigned to car (yellow)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, 'Unassigned', (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Display frame info
        cv2.putText(frame, f'Frame: {frame_nmr}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Vehicles: {len(track_ids)}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('License Plate Recognition', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # If frame reading failed, break the loop
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# write results
write_csv(results, './test.csv')
print(f"\nProcessing complete! Results saved to test.csv")
print(f"Total frames processed: {frame_nmr + 1}")
print(f"Total vehicles tracked: {len(set([car_id for frame_data in results.values() for car_id in frame_data.keys()]))}")