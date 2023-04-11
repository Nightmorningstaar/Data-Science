import cv2 as cv


# Works with images and videos
def rescaleFrame(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    length = int(frame.shape[0] * scale)

    return cv.resize(frame, (width, length), interpolation = cv.INTER_AREA)

# Works with live web cam
def changeRes(width, length):
        pass

capture = cv.VideoCapture("Videos/dog.mp4")

while True:
     isTrue, frame = capture.read()
     frame_resized = rescaleFrame(frame)
     cv.imshow("org", frame)
     cv.imshow("resize", frame_resized)

     if cv.waitKey(20) & 0xFF == ord("d"):
        break

capture.release()
cv.destroyAllWindows()

