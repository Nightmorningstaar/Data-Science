import os
import numpy as np
import cv2

people = []
for i in os.listdir(r"C:\python_prac\OpenCV\Faces\train"):
    people.append(i)

print(people)

dir = r"C:\python_prac\OpenCV\Faces\train"

features = []
labels = []

def create_train():
    for p in people:
        path = os.path.join(dir, p)
        label = people.index(p)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

            # Load the model for to detect the faces
            face_cascade = cv2.CascadeClassifier(r"Data\model_harcascade\face_detect_model.xml")

            face_rect = face_cascade.detectMultiScale(gray, 1.9, 1)
            
            for (x, y, w, h) in face_rect:
                face_roi = gray[y : y + h, x : x + w]
                features.append(face_roi)
                labels.append(label)
        

create_train()

## Before Use the recognizer model you have to convert labels and features into numpy array
features = np.array(features, dtype="object")
labels = np.array(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the face recognizer model using features and labels list
face_recognizer.train(features, labels)

np.save("features.npy", features)
np.save("labels.npy", labels)
face_recognizer.save("face_recog_trained.yml")
