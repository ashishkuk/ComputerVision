from ultralytics import YOLO
import cv2
import math
import cvzone

# Initialize video capture
#cap = cv2.VideoCapture(0) #for webcame 
#cap.set(3, 1280)  # Set width
#cap.set(4, 720)   # Set height
cap = cv2.VideoCapture("C:/Users/User/Desktop/Needed/ObjectDetection/objectdetection_yolo/videos/bikes.mp4")


# Load the YOLO model
model = YOLO("C:/Users/User/Desktop/Needed/ObjectDetection/objectdetection_yolo/yolo_weights/yolov8n.pt")

coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", 
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",`qqqqqq` "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", 
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
# based on coco dataset: common objects in context

while True:
    success, img = cap.read()  # success stores boolean and img stores image data
    results = model(img, stream=True)
    
    """
    The stream=True argument allows for processing the image in real-time, optimizing performance by handling frames one at a time instead of processing the entire batch.
results is a generator object that contains detection results for the image.Each r object contains information about the detected objects, including their bounding boxes, classes, and confidence scores.
boxes is an attribute of the result that holds the detected bounding boxes for the objects found in the image.

For each bounding box (box), you extract its coordinates using box.xyxy[0].
The xyxy attribute returns the bounding box coordinates in the format (x1, y1, x2, y2):
x1, y1: The coordinates of the top-left corner of the bounding box.
x2, y2: The coordinates of the bottom-right corner of the bounding box.The cv2.rectangle() function is used to draw the bounding box on the image.

The parameters specify:
The image on which to draw (img).
The top-left corner (x1, y1) and bottom-right corner (x2, y2).
The color of the rectangle (in BGR format, so (255, 0, 255) is purple).
The thickness of the rectangle border (3 pixels)."""
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)  # Fix the print statement to show all coordinates

            # Corrected rectangle drawing
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)  # Fixed the call here
            
            conf=math.ceil((box.conf[0]*100))/100  # upto two decimal places
            print(conf)
            
            cls=int(box.cls[0])
            
            
            cvzone.putTextRect(img,f'{coco_classes[cls]} {conf}',(max(0,x1),max(35,y1)),scale=0.7,thickness=1)  # take max of 0 and x,y , so if it goes in negative , the value doesnt go outside frame
            
            

    cv2.imshow("Image", img)  # Show the image
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
