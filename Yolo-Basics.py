from ultralytics import YOLO
import cv2

model=YOLO('pythonProject\Yolo-Weights\yolov8x.pt')
results=model("pythonProject\Images\image3.jpg",show=True)
cv2.waitKey(0)