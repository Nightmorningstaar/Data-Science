import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("Photos/cats.jpg")
cv2.imshow("cat", img)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# gray_hist = cv2.calcHist([gray], [0], None,histSize=[256], ranges=[0, 256])

# cv2.imshow("gray",gray)

# plt.figure()
# plt.xlabel("pixles range")
# plt.ylabel("# of pixels")
# plt.plot(gray_hist)
# plt.show()


### Get the pixle histogram for masked image

# blank = np.zeros(img.shape[:2], dtype= "uint8")

# circle =  cv2.circle(blank.copy(), (img.shape[1] // 2, img.shape[0] // 2), 50, 255, -1)
# cv2.imshow("circle", circle)

# masked_img = cv2.bitwise_and(gray, gray, mask = circle)
# cv2.imshow("masked_img", masked_img)


# gray_masked = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
# gray_hist_masked = cv2.calcHist([gray_masked], [0], None,histSize=[256], ranges=[0, 256])

# gray_hist_mask = cv2.calcHist([gray], [0], masked_img, [256], [0, 256])
# plt.figure()
# plt.xlabel("pixles range")
# plt.ylabel("# of pixels")
# plt.plot(gray_hist_mask)
# plt.show()


### Histogram for the colour image
color = ("b", "g", "r")

blank = np.zeros(img.shape[:2], dtype= "uint8")

circle =  cv2.circle(blank.copy(), (img.shape[1] // 2, img.shape[0] // 2), 50, 255, -1)
cv2.imshow("circle", circle)

masked_img = cv2.bitwise_and(img, img, mask = circle)
cv2.imshow("masked_img", masked_img)
plt.figure(figsize=(12, 8))
for i, col in enumerate(color):
    color_hist = cv2.calcHist([img], [i], circle, [256], [0, 256])
    
    plt.xlabel("pixles range")
    plt.ylabel("# of pixels")
    plt.plot(color_hist, color = col)

plt.show()
cv2.waitKey(0)