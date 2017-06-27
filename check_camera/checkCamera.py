import cv2
import sys

video_capture = cv2.VideoCapture(-1)
# video_capture.open(0)
print (video_capture.isOpened())

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if frame is None:
        continue

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == 27:
        break # esc to quit

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
