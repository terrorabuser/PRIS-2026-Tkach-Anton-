from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
# license_plate_detector = YOLO("license_plate_detector.pt")  # Раскомментируйте, когда модель будет доступна


cap = cv2.VideoCapture("data/raw/video_1.mp4")
ret = True

frame_nmbr = 0

while ret:
    frame_nmbr += 1
    ret, frame = cap.read()
    if ret and frame_nmbr > 10:
        break
    #Детект объектов
    detections = model(frame)[0]
    print(detections)