import cv2
import numpy as np

img = cv2.imread("Photos/cats.jpg")
cv2.imshow("cat", img)

# Averaging is technique in which we average out the surrounding pixles and the result will assign to the middle pixels
avg = cv2.blur(img, (3,3))
cv2.imshow("avg", avg)

# Gaussian blur is same as average blur but the only difference is this technique assign some weights and then do their calculations
gauss = cv2.GaussianBlur(img, (3,3), 0)
cv2.imshow("gauss", gauss)

#Median Blur is take the median value and it is also used to removing the noise in the image while bluring
medi = cv2.medianBlur(img, 3)
cv2.imshow("median", medi)

# Bilateral is the techniuqe in which blur the image keep edges as it is 
# sigmacolor = color is considered in the neighbourhood when blur is computed
# sigmaspace =  space is considered in the neighbourhood when blur is computed
bilet = cv2.bilateralFilter(img, 10, 15, 15)
cv2.imshow("bilateral", bilet)
cv2.waitKey(0)