from ultralytics import YOLO
import cv2


model=YOLO('C:/Users/User/Desktop/Needed/ObjectDetection/objectdetection_yolo/yolo_weights/yolov8n.pt') #will download on first run #n , m ,l for nano, medium , large , 
results=model("C:/Users/User/Desktop/Needed/ObjectDetection/objectdetection_yolo/images/motorcycle.jpg",show=True)

cv2.waitKey(0)