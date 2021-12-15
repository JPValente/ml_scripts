#!/usr/bin/env python3

import cv2                                              # OpenCV - computer vision libraries
import numpy as np                                      # NumPy - fancy math
import os                                               # Operating System IF (for access to path)
from matplotlib import pyplot as plt                    # PyPlot from MatPlotLib
import time                                             # Time functions
import mediapipe as mp                                  # MediaPipe - hand & pose tracking
from tensorflow.keras.utils import to_categorical       # TF categorical
from tensorflow.keras.models import Sequential          # TF Sequential models
from tensorflow.keras.models import load_model          # TF Load the saved model
from tensorflow.keras.layers import LSTM, Dense         # TF LSTM & Dense layers
from tensorflow.keras.callbacks import TensorBoard      # TF TensorBoard callback for logs
from sklearn.model_selection import train_test_split    # SciKit Learn
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score # SciKit metrics

# Define MediaPipe holistic model - we need more than just hand tracking (arm movement yay!) and drawing utils
mpHolisticModel = mp.solutions.holistic
mpDrawingUtils  = mp.solutions.drawing_utils

# Path definition for exported MediaPipe data
DATA_PATH = os.path.join('MP_Data')

# ASL Signs to detect - make an array of all signs to teach the model (['hello', 'thanks', etc...])
signs = np.array(['hello', 'yes', 'no', 'thanks', 'class'])

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

# Visualize probability calcs
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
def probViz(res, signs, inputFrame, colors):
    outputFrame = inputFrame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(outputFrame, (0, 60 + num*40), (int(prob*100), 90 + num*40), colors[num], -1)
        cv2.putText(outputFrame, signs[num], (0, 85 + num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return outputFrame

# Set up model for loading
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(signs.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights('SignModel.h5')
#model = load_model('SignModel')

# Sequence length (in frames) default 30 frames/second
seqLength = 30

# Map labels for each sign - loop through the MP_Data directory and parse each sign
labelMap = {label:num for num, label in enumerate(signs)}

sequences, labels = [], []
for sign in signs:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, sign))).astype(int):
        window = []
        for frameNum in range(seqLength):
            res = np.load(os.path.join(DATA_PATH, sign, str(sequence), "{}.npy".format(frameNum)))
            window.append(res)
        sequences.append(window)
        labels.append(labelMap[sign])

# Train/Test split
seqX = np.array(sequences)
labY = to_categorical(labels).astype(int)
xTrain, xTest, yTrain, yTest = train_test_split(seqX, labY, test_size = 0.05)

# Confustion matrix - debugging
yHat = model.predict(xTest)
yTrue = np.argmax(yTest, axis = 1).tolist()
yHat = np.argmax(yHat, axis = 1).tolist()
print(multilabel_confusion_matrix(yTrue, yHat))
print(accuracy_score(yTrue, yHat))

# Detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

# Video Detection Loop
# VideoCapture(0) *should* be the default webcam, this may vary by device or if multiple cams are attached
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Uncomment the below four lines to set a custom resolution (default appears to be 640x480?
#resHorz = 800	# 800 x 600 px
#resVert = 600	# 800 x 600 px
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, resHorz)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resVert)
with mpHolisticModel.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():

        # Read in video from webcam
        ret, frame = cap.read()

        # Process detections
        image, results = DetectionTracking(frame, holistic)

        # Draw computed landmarks
        DrawStyledLandmarks(image, results)

        # Make prediction to compare against model
        keypoints = ExtractKeypoints(results)
        sequence.insert(0, keypoints)
        sequence = sequence[:30]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis = 0))[0]
            print(signs[np.argmax(res)])
            predictions.append(np.argmax(res))

            # Visualization
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if signs[np.argmax(res)] != sentence[-1]:
                            sentence.append(signs[np.argmax(res)])
                    else:
                        sentence.append(signs[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Visualize probabilities   --- BROKEN RIGHT NOW DO NOT USE
            #image = probViz(res, signs, image, colors)

        cv2.rectangle(image, (0, 0), (640, 40), (45, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                       cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show image to screen
        cv2.imshow('OpenCV Webcam Feed - Press q to close', image)

        # Allow for graceful break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
