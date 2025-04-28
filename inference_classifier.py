# inference_classifier.py

import pickle
import cv2
import numpy as np

# 1) Load your saved dict and extract the real estimator
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']    # <— now `model` is a sklearn estimator

def predict(frame):
    """
    Take a BGR OpenCV frame, extract the largest contour ROI,
    preprocess to 28×28 binary image, and return the predicted label.
    """
    # Grayscale & threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 127, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find contours, pick the biggest
    cnts, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not cnts:
        return ''  # nothing detected

    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    if w < 10 or h < 10:
        return ''  # too small

    # Crop, resize, flatten & normalize
    roi = thresh[y:y+h, x:x+w]
    roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    arr = roi_resized.flatten().reshape(1, -1).astype(np.float32) / 255.0

    # Predict with the sklearn model
    label = model.predict(arr)[0]
    return str(label)
