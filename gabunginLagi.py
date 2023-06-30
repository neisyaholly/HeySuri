import torch
from ultralytics import YOLO
import numpy as np
import cv2
import pandas
import time
import pyttsx3
import speech_recognition as sr

text_speech = pyttsx3.init()

# Inisialisasi recognizer
r = sr.Recognizer()

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

lst = []
lstfinal = []
cap = cv2.VideoCapture(0)
heysuri = "hey siri"

start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections
    results = model(frame)
    # cv2.imshow('YOLO', np.squeeze(results.render())) 
    
    df = results.pandas().xyxy[0]
    
    # Menggunakan mikrofon sebagai input
    with sr.Microphone() as source:
        # print("Silakan mulai berbicara...")
        audio = r.listen(source)

        try:
            # Menggunakan Google Speech Recognition untuk mengubah suara menjadi teks
            text = r.recognize_google(audio)
            # print("Anda mengatakan: " + text)
            if text.casefold() == heysuri.casefold() :
                for i in df['name']: # name->labels
                    print(i) #ini print
                    lst.append(i)
                text_speech.say("there is")
                for obj in lst :
                    answer = obj
                    text_speech.say(answer)
                    text_speech.runAndWait()
                lstfinal.append(lst.copy())
                lst.clear()
        except sr.UnknownValueError:
            print("Maaf, tidak bisa mengenali suara Anda.")
        except sr.RequestError as e:
            print("Error: " + str(e))
        
    # if (time.time() - start_time) % 7 < 0.2:
    #     for i in df['name']: # name->labels
    #         print(i) #ini print
    #         lst.append(i)
    #     text_speech.say("there is")
    #     for obj in lst :
    #         answer = obj
    #         text_speech.say(answer)
    #         text_speech.runAndWait()
    #     lstfinal.append(lst.copy())
    #     lst.clear()
        
    if time.time() - start_time > 14 :
        break
        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

print(lstfinal)
