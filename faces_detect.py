import numpy as np
import cv2
import pickle
from faces_train import train_model

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer.create(radius = 1, neighbors = 88, grid_x = 8, grid_y = 8)

try:
	recognizer.read("training_data/data.yml")
except:
	print("No trained data found, train existing data...")
	train_model()

labels = {"person_name": 1}
with open("training_data/labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9)
    name = ''
    conf = 0
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        id_, conf = recognizer.predict(roi_gray)
        if True:# if 0 <= conf <= 100:
            font = cv2.FONT_HERSHEY_TRIPLEX
            name = str(labels[id_]).title()
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y-10), font, 1, color, stroke, cv2.LINE_AA)

        img_item = "temp/my-image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)
        stroke = 3
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    
    return frame, name, conf

    
    # #display the resulting frame
    # cv2.imshow('frame', frame) #show all frames as video

    # #break button declaration
    # if cv2.waitKey(20) & 0xFF == ord('q'):
    #     break

# #when everything done, release the capture
# cap.release()
# cap.destroyAllWindows()
