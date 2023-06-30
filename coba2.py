import torch
from ultralytics import YOLO
import numpy as np
import cv2
import pandas

lst = []

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
# img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
# results = model(img)

# Results
# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

# df = results.pandas().xyxy[0]
    
# for i in df['name']: # name->labels
#     lst.append(i)

lst = []
cap = cv2.VideoCapture(0)
# model = YOLO('yolov8n.pt')
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections
    results = model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render())) 
    
    df = results.pandas().xyxy[0]
    
    for i in df['name']: # name->labels
        print(i)
        lst.append(i)
        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()