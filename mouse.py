import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2 as cv

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'gesture_recognizer.task'
base_options = BaseOptions(model_asset_path=model_path)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    cap.open()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    cv.imshow('pic', frame)
    cv.waitKey(5000)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
