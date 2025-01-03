from ultralytics import YOLO  # Import YOLO from the ultralytics package
import cv2  # Import OpenCV for video capture and image processing
import math  # Import math for rounding confidence values

# Initialize webcam
cap = cv2.VideoCapture(0)  # Open the webcam (0 is the default webcam)
cap.set(3, 1280)  # Set the frame width to 1280 pixels
cap.set(4, 720)   # Set the frame height to 720 pixels

# Load YOLO model (running on CPU only)
model = YOLO("pythonProject\Yolo-Weights\yolov8n.pt")  # Load the YOLO model (using pretrained weights)

# Class names for the COCO dataset (80 objects)
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

# Capture video from the webcam indefinitely
while True:
    success, img = cap.read()  # Capture one frame from the webcam
    if not success:  # If there's an issue with the capture, break the loop
        break

    # Resize the image to 640x640 (required for the YOLO model)
    img_resized = cv2.resize(img, (640, 640))

    # Run YOLO model on the resized image
    results = model(img_resized)  # Get the detection results

    frame_center = img_resized.shape[1] // 2  # Calculate the center of the frame (for left/right check)

    for r in results:  # Loop through each result (could be multiple objects detected)
        boxes = r.boxes  # Get bounding boxes for detected objects

        for box in boxes:  # Iterate over each box (each box corresponds to one object)
            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the top-left (x1, y1) and bottom-right (x2, y2) coordinates
            
            # Get confidence and class ID
            conf = math.ceil((box.conf[0] * 100))  # Get confidence score and convert to percentage
            class_id = int(box.cls[0])  # Get the class ID (what object it is)

            # Get the class name based on the class ID
            class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"

            # Determine the position of the object relative to the center
            object_center = (x1 + x2) // 2  # Calculate the center of the detected object
            if object_center < frame_center:  # If object center is to the left of frame center
                position = "Left"
            else:  # Otherwise, it's to the right of the frame center
                position = "Right"

            # Draw a rectangle around the detected object
            cv2.rectangle(img, (x1, y1), (x2, y2), (25, 200, 253), 2)  # Draw rectangle with a specific color

            # Prepare the text to display (class name, confidence, and position)
            text = f"{class_name} ({conf}%) - {position}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]  # Get the size of the text box
            text_bg_x2 = x1 + text_size[0] + 10  # Set background width based on text size

            # Draw a black rectangle as background for the text
            cv2.rectangle(img, (x1, y1 - 30), (text_bg_x2, y1), (0, 0, 0), -1)  # Black rectangle for text background

            # Put the text on the frame
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Display text

    # Show the frame with bounding boxes, text, and left/right info
    cv2.imshow("Image", img)

    # Check if 'q' is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()  # Close the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
