import cv2
import numpy as np


img = cv2.imread("Photos/cats.jpg")
cv2.imshow("cat", img)

print(img.shape)

blank = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
cv2.imshow("blank", blank)

mask = cv2.circle(blank.copy(), (img.shape[1] // 2, img.shape[0] // 2), 80, 255, -1,)
cv2.imshow("circle", mask)

# Use Logical operators for masking
bitwise_and = cv2.bitwise_and(img, img, mask = mask)
cv2.imshow("Masked Image", bitwise_and)
cv2.waitKey(0)