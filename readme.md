# BIOMETRICS – FACE DETECTION

#### AUTHORS: Kateřina Jandová and Carlos García

Face segmentation and recognition. For face detection and segmentation we will use the **Haar Cascade** method provided by the OpenCV library. For face recognition we will use the **LBPH Face Recognizer** in the OpenCV library as well.


# How to use it:

This project consists of two stages:

### Enrollment: 
for this purpose we use the script “face_recorder.py”, this script just turns on the camera and start recording photos into the folder “/images/”, then we can go to this folder, select the best images (around 10 or 15 images is enough per person) and copy them into the folder “/train_images/” where are included the images we will use when training. In this image enrollment process it is better having pictures with bright light and white flat background.

### Face Detection: 
here we use the script “face_recognizer.py”, this script search every picture into “/train_images/”, then segment the faces in this pictures and associate this faces according to the name of the file, creating a classifier (in this case using a Local Binary Patterns Histograms Face Recognizer) and trains it to learn the must important features in each face.
Once It finishes training it starts a matching process either by using the camera or by checking some other user images (not the same used for training) into the folder “/validate_images/”.
