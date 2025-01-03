import cv2
import numpy as np
from ultralytics import YOLO
from strongsort import StrongSORT
from strongsort.utils import get_color_for_id
import time

# Initialize YOLO model (replace 'yolov8n.pt' with your desired YOLOv8 model)
model = YOLO("Yolo-Weights/yolov8l.pt")

# Initialize StrongSORT tracker
tracker = StrongSORT()

# Video source (e.g., webcam or video file)
video_source = 0  # Use 0 for webcam, or replace with a video file path
cap = cv2.VideoCapture(video_source)

# Window name for displaying the video
window_name = "YOLO + StrongSORT Visualization"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Start FPS calculation
    start_time = time.time()

    # YOLO model inference
    results = model(frame)
    detections = []

    # Extract bounding box coordinates and other info from YOLO results
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()  # Bounding box coordinates
        conf = result.conf.cpu().item()  # Confidence score
        class_id = int(result.cls.cpu().item())  # Class ID
        class_name = model.names[class_id]  # Get class name from YOLO model
        w, h = x2 - x1, y2 - y1  # Width and height

        # Only keep high confidence detections
        if conf > 0.5:
            detections.append([x1, y1, x2, y2, conf])

    # Update StrongSORT tracker with YOLO detections (converting bounding boxes to format [x1, y1, x2, y2, confidence])
    tracker.update(np.array(detections))

    # Draw detections and tracking info
    for track in tracker.tracks:
        track_id = track[1]  # Get the track ID
        x1, y1, x2, y2 = track[2]  # Bounding box coordinates
        class_name = "Object"  # You can modify this to display actual class names if available
        color = get_color_for_id(track_id)  # Get color for this object ID

        # Calculate confidence from the track data (can be adjusted)
        confidence = round(track[4], 2)  # Track confidence

        # Draw bounding box and track ID with additional info
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"ID: {track_id} {class_name} Conf: {confidence}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Display FPS on the screen
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
