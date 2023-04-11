import cv2
import numpy as np

blank = np.zeros((400, 400), dtype="uint8")

rect = cv2.rectangle(blank.copy(), (30, 30), (300, 300), 255, -1)
cv2.imshow("rectangle", rect)


circle =  cv2.circle(blank.copy(), (200,200), 170, 255, -1)
cv2.imshow("circle", circle)

# # Bitwise AND (It gives the intersection of 2 regions)
bitwise_and = cv2.bitwise_and(rect, circle)
cv2.imshow("bitwise_and", bitwise_and)

# Bitwise OR (It gives the intersect as well as non-intersect image)
bitwise_or = cv2.bitwise_or(circle, rect)
cv2.imshow("bitwise_or", bitwise_or)

# XOR(It gives non-intersecting image)
bitwise_xor = cv2.bitwise_xor(circle, rect)
cv2.imshow("bitwise_xor", bitwise_xor)

# Bitwise NOT(it invert the binaru colour in the image)
bitwise_not = cv2.bitwise_not(circle)
cv2.imshow("bitwise_not", bitwise_not)
cv2.waitKey(0)
