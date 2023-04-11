import cv2

nero = cv2.imread(r"C:\python_prac\OpenCV\Photos\group 1.jpg")
cv2.imshow("nero", nero)

gray = cv2.cvtColor(nero, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

## link for download the models https://github.com/opencv/opencv/tree/4.x/data/
# Load the model
# But this model is too sensitive towards noise in the image
face_cascade = cv2.CascadeClassifier(r"Data\model_harcascade\face_detect_model.xml")

# Apply the model, it returns face cordiantes
face_rect = face_cascade.detectMultiScale(gray, 1.9, 1)

# check number of faces in the image
print(len(face_rect))

# draw a rect on the image by co-ordinates you have
for (x, y, w, h) in face_rect:
    cv2.rectangle(nero, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Detect faces", nero)
cv2.waitKey(0)