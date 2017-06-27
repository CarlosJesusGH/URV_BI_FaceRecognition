import cv2
import sys

label_name = sys.argv[1]
video_capture = cv2.VideoCapture(0)

print

# Create full screen window
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

for i in range(250):
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    # If camera not loaded, keep going
    if frame is None:
        continue
    start_frame = 50
    waiting_frames = 5
    if i % waiting_frames == 0 and i > start_frame:
        counter = (i - start_frame) / waiting_frames
        # Transform to gray scale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Save frame as image
        cv2.imwrite("./images/" + label_name + "_" + str(counter) + ".png", gray_frame)
        cv2.putText(frame, str(counter), (50,50), cv2.FONT_HERSHEY_TRIPLEX, 2, 255)
    # Display the resulting frame
    cv2.imshow("window", frame)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
