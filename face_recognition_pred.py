import os
import numpy as np
import cv2

people = []
for i in os.listdir(r"C:\python_prac\OpenCV\Faces\train"):
    people.append(i)

print(people)

features = np.load("features.npy", allow_pickle = True)
lables = np.load("labels.npy")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_recog_trained.yml")


img = cv2.imread(r"C:\python_prac\OpenCV\Faces\train\Jerry Seinfield\7.jpg")
cv2.imshow("nero", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

face_cascade = cv2.CascadeClassifier(r"Data\model_harcascade\face_detect_model.xml")

face_rect = face_cascade.detectMultiScale(gray, 1.9, 1)


for (x, y, w, h) in face_rect:
    face_roi = gray[y : y + h, x : x + w]
    label, confidence = face_recognizer.predict(face_roi)
    print(label)
    print("Confidence : ", confidence)
    cv2.putText(img, str(people[label]), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Ben", img)
cv2.waitKey(0)
