# gradients or edges are different in maths but in the programming both are the same thing
# There are several other methods to calculate edges in the image

import cv2
import numpy as np

img = cv2.imread("C:/python_prac/OpenCV/Photos/park.jpg")
cv2.imshow("cats", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


### 1. Laplacian 
lap = cv2.Laplacian(gray, cv2.CV_64F)
print(lap.dtype)
# lap contains the neg value so first we convert them into positive value and the we have to 
# convert the the lap(float64) into image type that is np.uint8
lap_img = np.uint8(abs(lap))
print(lap_img)
cv2.imshow("Laplacin", lap_img)


### 2. Sobel

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
comb_sobel = cv2.bitwise_or(sobel_x, sobel_y)
cv2.imshow("sobel_x", sobel_x)
cv2.imshow("sobel_y", sobel_y)
cv2.imshow("Combined Image", comb_sobel)



### 3. Canny
canny = cv2.Canny(gray, 150, 170)
cv2.imshow("canny", canny)
cv2.waitKey(0)