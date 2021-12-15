#!/usr/bin/env python3

# Some code snippets from Nicholas Renotte

import numpy as np                                      # NumPy - fancy math
import os                                               # Operating System IF (for access to path)
from sklearn.model_selection import train_test_split    # SciKit Learn
from tensorflow.keras.utils import to_categorical       # TF categorical
from tensorflow.keras.models import Sequential          # TF Sequential models
from tensorflow.keras.layers import LSTM, Dense         # TF LSTM & Dense layers
from tensorflow.keras.callbacks import TensorBoard      # TF TensorBoard callback for logs

# Path definition for exported MediaPipe data
DATA_PATH = os.path.join('MP_Data')

# ASL Signs to detect - make an array of all signs to teach the model (['hello', 'goodbye', etc...])
signs = np.array(['hello', 'yes', 'no', 'thanks', 'class'])

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

# Allow logging during learning to help with debugging or general information
logDir = os.path.join('Logs')
tbCallback = TensorBoard(log_dir = logDir)

# Set up sequential model then begin compilation and fitting
model = Sequential()
model.add(LSTM(64, return_sequences = True, activation = 'relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences = True, activation = 'relu'))
model.add(LSTM(64, return_sequences = False, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(signs.shape[0], activation = 'softmax'))

model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
model.fit(xTrain, yTrain, epochs = 500, callbacks = [tbCallback])

model.summary()

# Model predictions
#res = model.predict(xTest)
#signs[np.argmax(res[1])]
#signs[np.argmax(yTest[1])]


# Save model weights
#model.save('SignModel')
model.save('SignModel.h5')
