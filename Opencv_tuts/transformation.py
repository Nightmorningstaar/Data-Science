import cv2
import numpy as np

img = cv2.imread("Photos/park.jpg")
cv2.imshow("park", img)

cv2.waitKey(0)

def transformation(img, x, y):
    transmat = np.float32([[1, 0, x], 
                            [0, 1, y]])

    return cv2.warpAffine(img, transmat, (img.shape[1], img.shape[0]))

# -x = left
# x = right
# -y = up
# y = down

transform = transformation(img, 100, 100)       
cv2.imshow("tranform", transform)


def rotate(img, angle, rotpoint = None):
    h, w = img.shape[:2]
    
    if rotpoint == None:
        x, y = h // 2, w // 2

    rotmat = cv2.getRotationMatrix2D((x, y), angle, 1.0)

    return cv2.warpAffine(img, rotmat, (img.shape[1], img.shape[0]))


rot_img = rotate(img, 45)
cv2.imshow("rot_img", rot_img)


# Flip the image
# 0 =  via x axis , 1 = via y axis -1 =  both x and y axis
flipped = cv2.flip(img, 0)
cv2.imshow("flip", flipped)
cv2.waitKey(0)
