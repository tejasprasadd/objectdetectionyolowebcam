from ultralytics import YOLO  # Import YOLO model from ultralytics
import cv2  # Import OpenCV for video capturing and drawing
import math  # Import math for calculations

# Initialize webcam (this will be replaced by the vehicle's camera in the actual EV)
cap = cv2.VideoCapture(0)  # Capture video from the default webcam
cap.set(3, 1280)  # Set frame width
cap.set(4, 720)   # Set frame height

# Load the YOLOv8n model (lightweight version for faster inference)
model = YOLO("pythonProject/Yolo-Weights/yolov8n.pt")  # Lightweight model for performance

# COCO dataset class names (80 object categories)
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Function to estimate distance (simplified based on object height)
def estimate_distance(bbox_height, frame_height):
    # Estimate distance based on the size of the bounding box in relation to the frame height
    return round((frame_height / bbox_height), 2) if bbox_height > 0 else 0

# Start real-time video capture and object detection
while True:
    success, img = cap.read()  # Read frame from the webcam
    if not success:
        break  # Exit loop if the frame could not be read

    # Resize the image for YOLO processing (faster inference with 640x640)
    img_resized = cv2.resize(img, (640, 640))

    # Run YOLO model on the frame
    results = model(img_resized)

    frame_height = img.shape[0]  # Height of the frame (for distance estimation)

    # Process each detection result
    for r in results:
        for box in r.boxes:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_height = y2 - y1  # Height of the bounding box (for distance estimation)

            # Get the confidence and class ID
            conf = math.ceil(box.conf[0] * 100)  # Confidence percentage
            class_id = int(box.cls[0])  # Class ID

            # Get the class name (object detected)
            class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"

            # Estimate the distance of the object
            distance = estimate_distance(bbox_height, frame_height)

            # Prepare text with class, confidence, and estimated distance
            label = f"{class_name} ({conf}%) - Distance: {distance} units"

            # Draw a bounding box around the detected object
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put the label with object class, confidence, and distance
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the image with bounding boxes, class names, and distance info
    cv2.imshow("YOLO Object Detection with Distance Estimation", img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
