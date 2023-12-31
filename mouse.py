import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2 as cv
import pyautogui as pygui

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

pygui.PAUSE = 0

cap = cv.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    x, y = pygui.position()
    success, image = cap.read()
    
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        # Index pointer
        x = hand_landmarks.landmark[8].x * IMAGE_WIDTH
        y = hand_landmarks.landmark[8].y * IMAGE_HEIGHT
        z = hand_landmarks.landmark[8].z
        # cv.circle(image,(int(x),int(y)), 100, (0,0,255), -1)
        disp_width = 540 - 80
        disp_height = 320 - 0
        if x >= 80 and x <= 540 and y >= 0 and y <= 320:
          x -= 80

          new_x = 1920 - x * 2000 / disp_width
          new_y = y * 1100 / disp_height
          print(1920 - new_x, new_y)
          pygui.moveTo(new_x, new_y, 0)

        # Visualizing landmarks
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
    # Display the box we have to stay in
    cv.rectangle(image,(540,0),(80,320),(0,255,0),3)

    # Flip the image horizontally for a selfie-view display.
    cv.imshow('MediaPipe Hands', cv.flip(image, 1))
    if cv.waitKey(5) & 0xFF == 27:
      break
cap.release()
    
