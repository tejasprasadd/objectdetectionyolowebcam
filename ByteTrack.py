import cv2
import numpy as np
import random
import time
from ultralytics import YOLO

# Initialize YOLO model (replace 'yolov8n.pt' with your desired YOLOv8 model)
model = YOLO("Yolo-Weights/yolov8n.pt")

# ByteTracker class
class ByteTracker:
    def __init__(self, iou_threshold=0.5, confidence_threshold=0.5):
        self.tracks = {}  # {track_id: {'bbox': [x, y, w, h], 'class': class_name, 'color': (r, g, b)}}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

    def update(self, detections):
        # Detections format: [x, y, w, h, confidence, class_name]
        high_conf_detections = [det for det in detections if det[4] >= self.confidence_threshold]

        updated_tracks = {}
        for det in high_conf_detections:
            matched = False
            for track_id, data in self.tracks.items():
                iou = self._iou(data['bbox'], det[:4])
                if iou > self.iou_threshold:
                    updated_tracks[track_id] = {
                        'bbox': det[:4],
                        'class': det[5],
                        'color': data['color']
                    }
                    matched = True
                    break
            if not matched:  # New track
                updated_tracks[self.next_id] = {
                    'bbox': det[:4],
                    'class': det[5],
                    'color': self._random_color()
                }
                self.next_id += 1

        self.tracks = updated_tracks
        return self.tracks

    def _iou(self, boxA, boxB):
        # Calculate Intersection over Union (IoU)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        return interArea / float(boxAArea + boxBArea - interArea)

    def _random_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Video source (e.g., webcam or video file)
video_source = 0  # Use 0 for webcam, or replace with a video file path
cap = cv2.VideoCapture(video_source)

tracker = ByteTracker()
window_name = "YOLO + ByteTrack Visualization"

prev_time = 0  # Store previous time for FPS calculation

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO model inference
    results = model(frame)
    detections = []

    for result in results[0].boxes:  # Loop through detected boxes
        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()  # Bounding box coordinates
        conf = result.conf.cpu().item()  # Confidence score
        class_id = int(result.cls.cpu().item())  # Class ID
        class_name = model.names[class_id]  # Get class name from YOLO model
        w, h = x2 - x1, y2 - y1  # Width and height
        detections.append([int(x1), int(y1), int(w), int(h), conf, class_name])

    # Update tracker with YOLO detections
    tracked_objects = tracker.update(detections)

    # Draw detections and tracking info
    for obj_id, data in tracked_objects.items():
        x, y, w, h = data['bbox']
        class_name = data['class']
        color = data['color']
        conf = max([det[4] for det in detections if det[:4] == data['bbox']])

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Prepare label with ID, class, and confidence
        label = f"ID: {obj_id} {class_name} {conf:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
