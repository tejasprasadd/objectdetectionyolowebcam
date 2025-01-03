import cv2
import math
from ultralytics import YOLO

# Constants
KNOWN_HEIGHT = 1.5  # Known height of the object (e.g., average car height in meters)
FOCAL_LENGTH = 800  # Estimated focal length in pixels (from camera calibration)

# Initialize camera and YOLO model
cap = cv2.VideoCapture(0)
model = YOLO("pythonProject\Yolo-Weights\yolov8n.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Get detections
    
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_height = y2 - y1

            # Class ID and confidence
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = math.ceil((box.conf[0] * 100))

            # Calculate distance in meters
            distance = round((KNOWN_HEIGHT * FOCAL_LENGTH) / bbox_height, 2)

            # Display bounding box, class name, and distance on the image
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{class_name} {conf}% Distance: {distance}m"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the image
    cv2.imshow("Distance Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
