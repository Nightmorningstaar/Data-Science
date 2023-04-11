import cv2

# Reading the image
img = cv2.imread("Photos/park.jpg")
cv2.imshow("park", img)


# Converting it into Gray-Scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)


# Blur the image
# If we increase the kernel size blur in the image also increases
blur = cv2.GaussianBlur(img, (7, 7), cv2.BORDER_DEFAULT)
cv2.imshow("blur", blur)

# Edges in the image
# Canny Edge Detection is a popular edge detection algorithm. It was developed by John F. Canny in
edges = cv2.Canny(img, 150, 200)
cv2.imshow("edges", edges)


## https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html

# Dailate in the images
dailate = cv2.dilate(edges, (3,3), iterations = 3)
cv2.imshow("dailate", dailate)


# Erode in the image
erode = cv2.erode(dailate, (3,3), iterations=3)
cv2.imshow("erode", erode)


# Resize
# Decrease quality from the orginal size use cv2.INTER_AREA
# Increase quality from the original size use INTER_CUBIC
resize = cv2.resize(img, (500, 500), interpolation = cv2.INTER_AREA)
cv2.imshow("lower_resize", resize)

resize2 = cv2.resize(img, (2400, 1600), interpolation = cv2.INTER_CUBIC)
cv2.imshow("increase_resize", resize2)


# Cropping the image
cropped = img[0:300, 300:400]
cv2.imshow("cropped", cropped)
cv2.waitKey(0)