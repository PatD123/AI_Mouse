import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import math
import cv2 as cv
import pyautogui as pygui

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

pygui.PAUSE = 0
pygui.MINIMUM_DURATION = 0.01
pygui.MINIMUM_SLEEP = 0.01

prev_px = 0
prev_py = 0
smoothing = 3

def rescale(x, y):
    disp_width = 540 - 80
    disp_height = 320 - 0

    x -= 80
    new_x = 1920 - x * 2000 / disp_width
    new_y = y * 1100 / disp_height
    return new_x, new_y

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

        # Thumb pointer
        t_x = hand_landmarks.landmark[4].x * IMAGE_WIDTH
        t_y = hand_landmarks.landmark[4].y * IMAGE_HEIGHT

        # Bottom index
        b_x = hand_landmarks.landmark[6].x * IMAGE_WIDTH
        b_y = hand_landmarks.landmark[6].y * IMAGE_HEIGHT

        # Index pointer
        p_x = hand_landmarks.landmark[8].x * IMAGE_WIDTH
        p_y = hand_landmarks.landmark[8].y * IMAGE_HEIGHT
        p_z = hand_landmarks.landmark[8].z

        # cv.circle(image,(int(x),int(y)), 100, (0,0,255), -1)

        if p_x >= 50 and p_x <= 580 and p_y >= 0 and p_y <= 320:
            new_tx, new_ty = rescale(t_x, t_y)
            new_bx, new_by = rescale(b_x, b_y)
            new_px, new_py = rescale(p_x, p_y)

            new_px = prev_px + (new_px - prev_px) / smoothing
            new_py = prev_py + (new_py - prev_py) / smoothing
            prev_px = new_px
            prev_py = new_py
            pygui.moveTo(new_px, new_py, 0)

            #if abs(new_px - prev_px) >= 5 and abs(new_py - prev_py) >= 5:
            #    prev_px = new_px
            #    prev_py = new_py
            #    pygui.moveTo(new_px, new_py, 0)

            if math.dist([new_tx, new_ty], [new_bx, new_by]) < 100:
                pygui.click(button='left', clicks=1, interval=0.25)

        

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
    
