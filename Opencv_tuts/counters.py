# Countours are nothing just the curve that joins the points along the boundries
# In programming point view they are same but in mathematical way they are different

import cv2

img = cv2.imread("Photos/cats.jpg")
cv2.imshow("cats", img)

# Graying the images
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)


# finding the edges
edges = cv2.Canny(img, 150, 200)
cv2.imshow("edge", edges)


# Than using the edges image to find the countours cv2.CHAIN_APPROX_NONE = nothing gives all countours
# wheareas cv2.CHAIN_APPROX_SIMPLE = reduce the countours suppose there is a line so there is 2 countours
# heirarachay is nothing gives us the countour in herarical way

countour, heirarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
print(len(countour))

# If we blur the the image(cv2.GaussianBlur) the number of countour also reduced

import numpy as np
blank = np.zeros((img.shape), dtype="utfin8")
draw_countour = cv2.drawContours(blank, countour, -1, (255, 0, 0))
cv2.imshow("draw_countour", draw_countour)

cv2.waitKey(0)