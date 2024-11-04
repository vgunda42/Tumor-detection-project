# Gesture to Speech Recognition System

## Overview

The Gesture to Speech Recognition System is an innovative application designed to enable hands-free communication using natural hand gestures. This project utilizes computer vision and deep learning techniques to recognize specific hand gestures and convert them into speech commands. The application enhances accessibility and convenience by allowing users to interact with others.


## Features

- Real-time hand gesture recognition using webcam input.
- Conversion of recognized gestures into corresponding speech outputs.
- User-friendly interface for easy interaction.
- Support for multiple gestures like "FINE", "WATER", "LIGHT ON", and more.


## Prerequisites

- Python 
- TensorFlow
- OpenCV
- Streamlit
- pyttsx3

## How it Works

1. **Video Capture:** The application captures video frames from the user's webcam.
2. **Region of Interest (ROI) Extraction:** A specific region of the frame is extracted, focusing on the hand gesture.
3. **Image Preprocessing:** The extracted ROI is preprocessed to prepare it for model input.
4. **Model Prediction:** The preprocessed image is fed into the trained machine learning model, which predicts the most likely gesture.
5. **Text-to-Speech Conversion:** The predicted gesture is converted into text, and then the text is converted into speech using the pyttsx3 library.
6. **Display and Speech Output:** The predicted gesture and corresponding speech output are displayed on the web interface.

