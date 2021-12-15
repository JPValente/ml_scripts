#!/usr/bin/env python3

# Some code snippets from Nicholas Renotte

import cv2                              # OpenCV - computer vision libraries
import numpy as np                      # NumPy - fancy math
from matplotlib import pyplot as plt    # PyPlot from MatPlotLib
import time                             # Time functions
import mediapipe as mp                  # MediaPipe - hand & pose tracking

# Define MediaPipe holistic model - we need more than just hand tracking (arm movement yay!) and drawing utils
mpHolisticModel = mp.solutions.holistic
mpDrawingUtils  = mp.solutions.drawing_utils

# Detection Tracking Process Function
# args: image - the captured frame
#       model - the holistic model being generated
def DetectionTracking(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert captured image from BlueGreenRed to RedGreenBlue
    image.flags.writeable = False                   # Prevent writing to the current frame
    results = model.process(image)                  # Moke a prediction and save that to results
    image.flags.writeable = True                    # Re-enable writing to the frame
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Re-convert image back to BlueGreenRed
    return image, results

# Draw CV landmarks to the frame
def DrawLandmarks(image, results):
    mpDrawingUtils.draw_landmarks(image, results.face_landmarks, mpHolisticModel.FACE_CONNECTIONS)          # Draw face connections from MediaPipe
    mpDrawingUtils.draw_landmarks(image, results.pose_landmarks, mpHolisticModel.POSE_CONNECTIONS)          # Draw Pose connections from MediaPipe
    mpDrawingUtils.draw_landmarks(image, results.left_hand_landmarks, mpHolisticModel.HAND_CONNECTIONS)     # Draw left hand connections from MediaPipe
    mpDrawingUtils.draw_landmarks(image, results.right_hand_landmarks, mpHolisticModel.HAND_CONNECTIONS)    # Draw right hand connections from MediaPipe

# Draw CV Landmarks (color) to the frame
def DrawStyledLandmarks(image, results):
    mpDrawingUtils.draw_landmarks(image, results.face_landmarks, mpHolisticModel.FACE_CONNECTIONS,
                                 mpDrawingUtils.DrawingSpec(color = (80, 110, 10), thickness = 1, circle_radius = 1),
                                 mpDrawingUtils.DrawingSpec(color = (80, 256, 121), thickness = 1, circle_radius = 1)
                                 )
    mpDrawingUtils.draw_landmarks(image, results.pose_landmarks, mpHolisticModel.POSE_CONNECTIONS,
                                 mpDrawingUtils.DrawingSpec(color = (80, 22, 10), thickness = 2, circle_radius = 4),
                                 mpDrawingUtils.DrawingSpec(color = (80, 44, 121), thickness = 2, circle_radius = 2)
                                 )
    mpDrawingUtils.draw_landmarks(image, results.left_hand_landmarks, mpHolisticModel.HAND_CONNECTIONS,
                                 mpDrawingUtils.DrawingSpec(color = (121, 22, 76), thickness = 2, circle_radius = 4),
                                 mpDrawingUtils.DrawingSpec(color = (121, 44, 250), thickness = 2, circle_radius = 2)
                                 )
    mpDrawingUtils.draw_landmarks(image, results.right_hand_landmarks, mpHolisticModel.HAND_CONNECTIONS,
                                 mpDrawingUtils.DrawingSpec(color = (245, 110, 65), thickness = 2, circle_radius = 4),
                                 mpDrawingUtils.DrawingSpec(color = (245, 65, 230), thickness = 2, circle_radius = 2)
                                 )

# Video Capture Loop
# VideoCapture(0) *should* be the default webcam, this may vary by device or if multiple cams are attached
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Uncomment the below four lines to set a custom resolution (default appears to be 640x480?
resHorz = 800  # 800 x 600 px
resVert = 600  # 800 x 600 px
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resHorz)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resVert)

# Set MediaPipe model as holistic type and begin detection loop
with mpHolisticModel.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():
         # Read feed from webcam; 2 return values https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
        ret, frame = cap.read()

        # Process detections
        image, results = DetectionTracking(frame, holistic)
        #print(results)

        # Draw computed landmarks and show frame
        DrawStyledLandmarks(image, results)
        cv2.imshow('OpenCV Webcam Feed - Press q to close', image)

        # Allow for graceful break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
