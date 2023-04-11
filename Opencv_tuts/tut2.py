import cv2 as cv

# img = cv.imread("Photos/cat_large.jpg")

# cv.imshow("cat_large", img)

# cv.waitKey(0)

## Reading Videos

# It is taking integer argumets(0,1,2,3) these args will used when you are using web camera or multiple cameras then using 0,1,2 etc 
# Otherwise give path of the videos

# This code gives an error when you run that is -215: Assertion Failed because after end of the video there is no
# frame in that particular location that is why its giving us the error

capture = cv.VideoCapture("Videos/dog.mp4")
while True:
    isTrue, frame = capture.read()
    cv.imshow("videos", frame)

    # Basically we waiting for 20 sec and when letter d is pressed stop playing the video
    if cv.waitKey(20) & 0xFF == ord("d"):
        break

capture.release()
cv.destroyAllWindows()

