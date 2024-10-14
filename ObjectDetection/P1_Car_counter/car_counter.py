

from ultralytics import YOLO
import cv2
import math
import cvzone
import numpy as np

from sort import *






# Initialize video capture
#cap = cv2.VideoCapture(0) #for webcame 
#cap.set(3, 1280)  # Set width
#cap.set(4, 720)   # Set height
cap = cv2.VideoCapture("C:/Users/User/Desktop/Needed/ObjectDetection/objectdetection_yolo/videos/cars.mp4")


# Load the YOLO model
model = YOLO("C:/Users/User/Desktop/Needed/ObjectDetection/objectdetection_yolo/yolo_weights/yolov8n.pt")

coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", 
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork","knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", 
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
# based on coco dataset: common objects in context


mask=cv2.imread("C:/Users/User/Desktop/Needed/ObjectDetection/P1_Car_counter/mask_2.png")

tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)

totalCount=[]

limits=[330,350,673,350]


while True:
    success, img = cap.read()  # success stores boolean and img stores image data
    imgRegion=cv2.bitwise_and(img,mask)
    results = model(imgRegion, stream=True)
    
    detections=np.empty((0,5))
    
    
    
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
            w = x2 - x1  # Width of the bounding box
            h = y2 - y1  # Height of the bounding box
            print(x1, y1, x2, y2)  # Fix the print statement to show all coordinates

            # Corrected rectangle drawing
            #cv2.rectangle(img, (x1, y1), (x2, y2),(255, 0, 255), 3)  # Fixed the call here  #cv2.rectangle(image, pt1, pt2, color, thickness, lineType, shift)

            
            conf=math.ceil((box.conf[0]*100))/100  # upto two decimal places
            print(conf)
            
            cls=int(box.cls[0])
            
            
            #cvzone.putTextRect(img,f'{coco_classes[cls]} {conf}',(max(0,x1),max(35,y1)),scale=0.7,thickness=1)  # take max of 0 and x,y , so if it goes in negative , the value doesnt go outside frame
            
            current_class=coco_classes[cls]
            
            if current_class in ["car", "truck", "bus", "motorbike"] and conf > 0.30:
                #cvzone.putTextRect(img, f'{current_class} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=3)  # Closing the function properly
                #cvzone.cornerRect(img, (x1, y1, w, h), l=9,rt=5)
                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))
                
                
    

    resultsTracker=tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
    
    
    
    for result in resultsTracker:
        
        x1,y1,x2,y2,id=result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        
        w,h= x2-x1 ,y2-y1
        
        cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=2,colorR=(255,0,0))
        
        cvzone.putTextRect(img, f'{current_class} {int(id)}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=3)

        cx,cy=x1+w/2,y1+h/2
        cv2.circle(img,(int(cx),int(cy)),5,(255,0,255),cv2.FILLED)   # as soon as circle(center), crosses the line , the count is registered
        
        
        if limits[0]<cx<limits[2] and limits[1]-15<cy<limits[1]+15:
            if totalCount.count(id) ==0:  # if id is not already present in list , append 
                totalCount.append(id)
                cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)
                
    cvzone.putTextRect(img, f'Count:{len(totalCount)}',(50,50))
    
            
        
            
    
    
            
        
            
            

    cv2.imshow("Image", img) 
    cv2.imshow("ImgRegion",imgRegion) # Show the image
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
        
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
