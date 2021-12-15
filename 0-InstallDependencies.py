#!/usr/bin/env python3

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Installs TensorFlow v2.4.1, TensorFlow (GPU) v2.4.1, opencv for Python, mediapipe, sklearn, and matplotlib
#get_ipython().system('pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python mediapipe sklearn matplotlib
install("tensorflow==2.4.1")
install("tensorflow-gpu==2.4.1")
install("opencv-python")
install("mediapipe")
install("sklearn")
install("matplotlib")
