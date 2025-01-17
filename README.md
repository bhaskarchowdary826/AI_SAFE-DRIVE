# AI Safe Drive 🚗

An AI-driven system that combines **emotion detection** and **sleep detection** to enhance driving safety and user experience. It plays emotion-based music and alerts drivers if signs of drowsiness are detected.

---

## Introduction
This project aims to improve driving safety by detecting **driver emotions** and **drowsiness** in real-time. Using a webcam feed, the system:
1. **Detects Emotions**: Classifies the driver's facial expression into one of seven emotions (angry, disgusted, fearful, happy, neutral, sad, surprised) and plays corresponding music.
2. **Detects Drowsiness**: Monitors eye movement and head position to detect signs of drowsiness and triggers an alert (e.g., horn sound) to wake the driver.

The system is built using **Python**, **OpenCV**, **Dlib**, and **TensorFlow/Keras**.

---

## Dependencies
To run this project, you need the following:
- Python 3
- OpenCV
- TensorFlow/Keras
- NumPy
- Dlib
- PyDub (for playing music)

---

## Installation
## Clone the Repository and Setup


git clone https://github.com/bhaskarchowdary826/AI-Safe-Drive.git
cd AI-Safe-Drive

   
---

## Download Required Files
Before running the project, you need the following files:

Haar Cascade Files:

Download haarcascade_frontalface_default.xml

Download haarcascade_eye.xml

Place these files inside the src folder. These files are used for detecting faces and eyes from the webcam feed.

## Pre-Trained Models:

Download the pre-trained emotion detection model (emotion_model.h5).

Download the pre-trained sleep detection model (sleep_model.h5).

Place these files inside the models folder.

---


## Data Preparation
Emotion Detection Dataset
The emotion detection model is trained on the FER-2013 dataset, which consists of 35,887 grayscale, 48x48-sized face images with seven emotions (angry, disgusted, fearful, happy, neutral, sad, surprised). You can download the dataset from Kaggle.

Sleep Detection Dataset
The sleep detection model uses custom data collected from webcam feeds. You can use your own dataset or collect data using the data_collection.py script provided in the src folder.

---
## Training the Models
Emotion Detection Model
To train the emotion detection model from scratch, run:


cd src
python emotion_detection.py --mode train
Sleep Detection Model




To train the sleep detection model from scratch, run:


cd src
python sleep_detection.py --mode train



---

## Running the System
Using Pre-Trained Models
If you prefer to use the pre-trained models, after downloading emotion_model.h5 and sleep_model.h5, place them in the models folder.

To run the AI Safe Drive system, use the following command:



cd src
python app.py



---


## Algorithm
Face Detection: The Haar Cascade method is used to detect faces in each frame from the webcam feed.

Emotion Classification: The region of the image containing the face is resized to 48x48 pixels and passed as input to the Convolutional Neural Network (CNN).

Sleep Detection: Eye movement and head position are monitored using OpenCV and Dlib.

Alert System: If drowsiness is detected, an alert (e.g., horn sound) is triggered.

---

## Accuracy
Emotion Detection: Achieves ~65% accuracy on the FER-2013 dataset.

Sleep Detection: Achieves ~90% accuracy in detecting drowsiness.

--

## References
FER-2013 Dataset: Kaggle

OpenCV Haar Cascades: GitHub

---

## Important Notes
Ensure you have the Haar Cascade files (haarcascade_frontalface_default.xml and haarcascade_eye.xml) in the src folder for face and eye detection.

Place the pre-trained models (emotion_model.h5 and sleep_model.h5) in the models folder if you are not training the models from scratch.

---

Connect with Me 🌐

