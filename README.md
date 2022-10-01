# Volleyball_Position_Detection_System
This repository is all about project on detection of volleyball's player positions using YOLOv3 object detection.
Here the project is completed in three steps:

1. Data Collection
For this project I have collected the datasets which is already annotated from this site: https://universe.roboflow.com/
From this site we can get already annotated datasets for image classification, segmentation and detection.

2. Training
In this project I have trained almost 1100 images with its annotation files
To train the images we have few steps:
a) Mount the drive
b) Clone the darknet from https://github.com/AlexeyAB/darknet.git. In this darknet directory we can get configuration and Makefiles of YOLO
which is further used in training process
c) Enabled the GPU , OPENCV and CUDNN in Makefile inside darknet
d) Copy the configuration files of YOLOv3 from darknet and customize the configuration files by changing the max_batches, steps and no.of filters and
save it inside darknet directory.
e) Download darknet53.conv.74 and keep it inside custom_weight directory.
f) Unzip the images and create classes.txt and classes.name file inside it which includes classes of images.
g) Create train.txt, test.txt and labelled_data.data files by running two files:
 i) https://github.com/lalchhabi/Volleyball_Position_Detection_System/blob/master/training/creating-files-data-and-name.py
 ii) https://github.com/lalchhabi/Volleyball_Position_Detection_System/blob/master/training/creating-train-and-test-txt-files.py
 
h) Create backup directory to save the trained models or weights.
i) Now Train the images.

link for training file: https://github.com/lalchhabi/Volleyball_Position_Detection_System/blob/master/training/volleyball_detection.ipynb

3. Testing
After training the images we can get trained models or weights in backup directory.
Now downlaod the trained weights in your local computer and configuration files and classes.txt as well.
After that run the https://github.com/lalchhabi/Volleyball_Position_Detection_System/blob/master/testing/testing.py file with some test images.

link for the trained models: https://drive.google.com/drive/folders/1qvy7dSfIgnHdGN4k7DCxdvJlLnOz2gTE?usp=sharing

This is the overall project description.

<img alt = 'coding' width = "1000" height = "500" src = "https://github.com/lalchhabi/Volleyball_Position_Detection_System/blob/master/testi/volleyball_test_result1.png">

<img alt = 'coding' width = "1000" height = "500" src = "https://github.com/lalchhabi/Volleyball_Position_Detection_System/blob/master/testing/volleyball_test_result2.png">

