from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import model_from_json
import torch
from ultralytics import YOLO
import numpy as np
import cv2
import pandas
import time
import pyttsx3
import speech_recognition as sr

app = Flask(__name__)

@app.route('/')
def index():
    text_speech = pyttsx3.init()

    # Inisialisasi recognizer
    r = sr.Recognizer()

    # Load YOLOv5 model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    lst = []
    lstfinal = []
    lstDistance = []
    cap = cv2.VideoCapture(0)
    heysuri = "hey siri"
    hisuri = "hi siri"

    # Parameters for distance calculation
    Known_distance = 18  # centimeters
    Known_width = 15  # centimeters
    Focal_length_found = 0  # Initialize focal length

    # Colors
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLACK = (0, 0, 0)

    # Fonts
    fonts = cv2.FONT_HERSHEY_SIMPLEX

    # Get focal length
    def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
        focal_length = (width_in_rf_image * measured_distance) / real_width
        return focal_length

    # Distance estimation
    def Distance_finder(Focal_Length, real_object_width, object_width_in_frame):
        distance = (real_object_width * Focal_Length) / object_width_in_frame
        return distance

    # Detect objects using YOLOv5
    def detect_objects(image):
        results = model(image)
        pred = results.pred[0]
        return pred

    # Read reference image and find object width in pixels
    ref_image = cv2.imread("buatref.jpg")
    pred_ref = detect_objects(ref_image)
    ref_object_width = pred_ref[:, 2] - pred_ref[:, 0]

    # Calculate average object width
    ref_object_width_avg = np.mean(ref_object_width.cpu().numpy())
    Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, ref_object_width_avg)
    print(Focal_length_found)

    # Initialize camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pred_frame = detect_objects(frame)
        results = model(frame)
        df = results.pandas().xyxy[0]

        with sr.Microphone() as source:
            # print("Silakan mulai berbicara...")
            audio = r.listen(source)

            try:
                # Menggunakan Google Speech Recognition untuk mengubah suara menjadi teks
                text = r.recognize_google(audio)
                print("Anda mengatakan: " + text)
                if text.casefold() == heysuri.casefold() or text.casefold() == hisuri.casefold():
                    for detection in pred_frame:
                        object_width = detection[2] - detection[0]
                        object_width_avg = np.mean(object_width.cpu().numpy())
                        distance = Distance_finder(Focal_length_found, Known_width, object_width_avg)

                        # Get coordinates for drawing the bounding box
                        x1, y1, x2, y2 = map(int, detection[:4])

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
                        
                        distance_text = f"{round(distance, 2)} CM"
                        lstDistance.append(round(distance, 2))
                        # Draw distance text above the bounding box
                        cv2.putText(frame, f"               {round(distance, 2)} CM", (x1, y1 - 10), fonts, 0.6, BLACK, 2)
                        # text_speech.say(f"The distance is {distance_text}")
                    
                    for i in df['name']: # name->labels
                        print(i) #ini print
                        lst.append(i)
                    text_speech.say("there is")
                    index = 0
                    for obj in lst :
                        answer = obj
                        text_speech.say(answer)
                        text_speech.say(f"with distance {lstDistance[index]} CM")
                        text_speech.runAndWait()
                        index+=1
                    lstfinal.append(lst.copy())
                    lst.clear()
                    lstDistance.clear()
                if text.casefold() == 'please stop' :
                    break
            except sr.UnknownValueError:
                print("Maaf, tidak bisa mengenali suara Anda.")
            except sr.RequestError as e:
                print("Error: " + str(e))
        
        # cv2.imshow("frame", frame)
        
        # cv2.imshow('YOLO', np.squeeze(results.render()))

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['name']
    
    exec(open('bismillah.py').read())
    
    return render_template('submit.html', name=name)

# if __name__ == '__main__':
app.run(debug=True)
