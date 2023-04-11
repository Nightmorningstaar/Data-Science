import cv2 as cv

# Read a image to convert it into pixels and matrix
img = cv.imread("Photos/cat.jpg")

# Pass the name of image and matrix i.e img
cv.imshow("Cat", img)

# Wait until I press any key
cv.waitKey(0)