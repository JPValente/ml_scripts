#!/usr/bin/env python3

# Some code snippets from Nicholas Renotte

import cv2                                              # OpenCV - computer vision libraries
import numpy as np                                      # NumPy - fancy math
import os                                               # Operating System IF (for access to path)
from matplotlib import pyplot as plt                    # PyPlot from MatPlotLib
import time                                             # Time functions
import mediapipe as mp                                  # MediaPipe - hand & pose tracking


# Define MediaPipe holistic model - we need more than just hand tracking (arm movement yay!) and drawing utils
mpHolisticModel = mp.solutions.holistic
mpDrawingUtils  = mp.solutions.drawing_utils

# Path definition for exporting MediaPipe data
DATA_PATH = os.path.join('MP_Data')

# ASL Signs to detect - make an array of all signs to teach the model (['hello', 'goodbye', etc...])
signs = np.array(['hello', 'yes', 'no', 'thanks', 'class'])

# Number of detection sequences to poll for training
numSequences = 4

# Sequence length (in frames) default 30 frames/second
seqLength = 30

# Folder number for collecting sequences
startFolder = 1

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

# Draw CV Landmarks (colored) to the frame - as above in DrawLandmarks, just with some flair
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

# Extract key points
def ExtractKeypoints(results):
    pose  = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face  = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lHand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rHand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lHand, rHand])

# Set up a directory for each sign being taugh to the ML model
for sign in signs:
    for sequence in range(1, numSequences + 1):
        try:
            os.makedirs(os.path.join(DATA_PATH, sign, str(sequence)))
        except:
            pass


# Video Capture Loop
# VideoCapture(0) *should* be the default webcam, this may vary by device or if multiple cams are attached
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Uncomment the below four lines to set a custom resolution (default appears to be 640x480?
#resHorz = 800  # 800 x 600 px
#resVert = 600  # 800 x 600 px
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, resHorz)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resVert)
with mpHolisticModel.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    # Loop through signs
    for sign in signs:
        # Loop through sequence video of each sign
        for sequence in range(startFolder, startFolder + numSequences):
            # Loop through each frame in the sequence
            for frameNum in range(seqLength):

                # Read feed from webcam; 2 return values https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
                ret, frame = cap.read()

                # Process detections
                image, results = DetectionTracking(frame, holistic)

                # Draw computed landmarks
                DrawStyledLandmarks(image, results)

                # Begin collecting keyframe data for signs
                # Wait 5 seconds to alert the user before recording to get into position
                if frameNum == 0:
                    cv2.putText(image, 'START COLLECTION', (120, 200),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting sequence for {} Video number {}'.format(sign, sequence), (15, 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Webcam Feed - Press q to close', image)
                    cv2.waitKey(500)
                else:
                    cv2.putText(image, 'Collecting sequence for {} Video number {}'.format(sign, sequence), (15, 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Webcam Feed - Press q to close', image)

                keypoints = ExtractKeypoints(results)
                npPath = os.path.join(DATA_PATH, sign, str(sequence), str(frameNum))
                np.save(npPath, keypoints)

                # Allow for graceful break
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

# Quick fix for some erratic closing behavior - sometimes window would not close at end of loop
cap.release()
cv2.destroyAllWindows()
