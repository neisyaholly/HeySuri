import torch
from ultralytics import YOLO
import numpy as np
import cv2
import pandas
import time

# lst = []

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

lst = []
lstfinal = []
cap = cv2.VideoCapture(0)
# model = YOLO('yolov8n.pt')

start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections
    results = model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render())) 
    
    df = results.pandas().xyxy[0]
    
    # for i in df['name']: # name->labels
    #     print(i) #ini print
    #     lst.append(i)
        
    if (time.time() - start_time) % 5 < 0.2:
        for i in df['name']: # name->labels
            print(i) #ini print
            lst.append(i)
        lstfinal.append(lst.copy())
        lst.clear()
        
    elif time.time() - start_time > 10 :
        break
        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

print(lstfinal)
