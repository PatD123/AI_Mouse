import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2 as cv

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'gesture_recognizer.task'
base_options = BaseOptions(model_asset_path=model_path)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        if(len(result.gestures) == 0):
            return
        
        output = output_image.numpy_view()

        landmarks_2d = landmark_pb2.NormalizedLandmarkList()
        for hand_landmark in result.hand_landmarks[0]:
            landmarks_2d.landmark.add(x=hand_landmark.x, y=hand_landmark.y)

        mp_drawing.draw_landmarks(
                    output,
                    landmarks_2d,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        cv.imshow("asfd", output)
        cv.waitKey(5000)

     

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    cap.open()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)
    with GestureRecognizer.create_from_options(options) as recognizer:
        recognizer.recognize_async(mp_image, cv.CAP_PROP_POS_MSEC)

cap.release()

    
