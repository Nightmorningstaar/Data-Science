import cv2
import numpy as np

# Reading the image
img = cv2.imread("Photos/park.jpg")
cv2.imshow("park", img)

b, g, r = cv2.split(img)
# cv2.imshow("blue", b)
# cv2.imshow("green", g)
# cv2.imshow("red", r)

print(b.shape, g.shape, r.shape)


blank = np.zeros(img.shape[:2], dtype="uint8")
print("blank", blank)

blue = cv2.merge([b, blank, blank])
green = cv2.merge([blank, g, blank])
red = cv2.merge([blank, blank, r])

cv2.imshow("blue", blue)
cv2.imshow("green", green)
cv2.imshow("red", red)
cv2.waitKey(0)
