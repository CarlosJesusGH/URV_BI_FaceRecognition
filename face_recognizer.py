#!/usr/bin/python

# OLD - Run docker container like this
# docker run --name opencv -v '/home/cj/Dropbox/Personal/Study/MasterDegreeArtificialIntelligence/1st Semester/BI/Project 4':/host_folder  --device=/dev/video0 --net=host -e DISPLAY -v /tmp/.X11-unix ibotdotout/python-opencv bash
# Run docker container like this
# docker stop opencv && docker rm opencv && docker run -d --name opencv -it -v '/home/cj/Dropbox/Personal/Study/MasterDegreeArtificialIntelligence/1st Semester/BI/Project 4':/host_folder  --device=/dev/video0 --net=host -e DISPLAY -v /tmp/.X11-unix ibotdotout/python-opencv && docker exec opencv apt install xauth -y && docker exec opencv xauth add $(xauth list) && docker attach opencv
# To solve docker image error
# ln /dev/null /dev/raw1394 && cd .. && cd host_folder && python face_recognizer.py




# Import the required modules
import cv2, os
import numpy as np
from PIL import Image

# cv2.__version__   # check OpenCV version
person = {}
person[1] = "cj"
person[2] = "albany"
person[3] = "jorge"
person[4] = "fadi"
person[5] = "saddam"
person[6] = "mikel"
person[7] = "kate"

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
# cascadePath = '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

# Eigenface Recognizer - createEigenFaceRecognizer()
# Fisherface Recognizer  - createFisherFaceRecognizer()
# Local Binary Patterns Histograms Face Recognizer - createLBPHFaceRecognizer()
# For face recognition we will the the LBPH Face Recognizer
recognizer = cv2.createLBPHFaceRecognizer()

def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    # image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image in grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        # nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        nbr = int(os.path.split(image_path)[1].split("_")[0].split("-")[1])
        # Detect the face in the image
        # faces = faceCascade.detectMultiScale(image)
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels

def predict_using_camera():
    # Show camera and identify
    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            predict_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
            cv2.putText(frame,person[nbr_predicted], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            print person[nbr_predicted] + ", " + str(conf)
        # Display the resulting frame
        cv2.imshow("window", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

def predict_using_pictures():
    # Append the images in validate_images into image_paths
    path = './validate_images'
    # image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad')]
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    for image_path in image_paths:
        predict_image_pil = Image.open(image_path).convert('L')
        predict_image = np.array(predict_image_pil, 'uint8')
        # faces = faceCascade.detectMultiScale(predict_image)
        faces = faceCascade.detectMultiScale(
            predict_image,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
            # nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
            nbr_actual = int(os.path.split(image_path)[1].split("_")[0].split("-")[1])
            if nbr_actual == nbr_predicted:
                print ("{} is Correctly Recognized with confidence {}".format(nbr_actual, conf))
            else:
                print ("{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted))
            cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
            cv2.waitKey(1000)

# Main
# Check if train file exists otherwise train
train_output_path = './train_output'
if os.path.exists(train_output_path):
    recognizer.load(train_output_path)
else:
    # Path to the Yale Dataset
    path = './train_images'
    # Call the get_images_and_labels function and get the face images and the
    # corresponding labels
    images, labels = get_images_and_labels(path)
    cv2.destroyAllWindows()
    # Perform the tranining
    recognizer.train(images, np.array(labels))
    recognizer.save("train_output")

predict_using_camera()
# predict_using_pictures()
