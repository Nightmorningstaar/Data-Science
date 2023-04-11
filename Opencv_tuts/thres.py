import cv2

img = cv2.imread("C:/python_prac/OpenCV/Photos/cats 2.jpg")
cv2.imshow("cats", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Simple Thresholding
threshold, thres_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

threshold_inv, inv_thres_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("Threshold image", thres_img)
cv2.imshow("Inv Threshold image", inv_thres_img)

# Adapative Thresholding : Sometimes simple thersholding works but in most of the cases it doesnt work
# so in Adapative Thresholding let computer decide the what is the best optimal value of thershold

adp_thres_img =  cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3) 
cv2.imshow("Adaptive Threshold image", adp_thres_img)

cv2.waitKey(0)