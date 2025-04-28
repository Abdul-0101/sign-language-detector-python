# inference_classifier.py

import cv2
import numpy as np
import pickle

# Load your trained model (make sure model.p is in the project root)
with open('model.p', 'rb') as f:
    model = pickle.load(f)

def predict(frame):
    """
    Take a BGR OpenCV frame, extract the largest contour ROI,
    preprocess to 28×28 binary image, and return the predicted label.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Threshold to get a binary image (you may need to tweak threshold)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Find contours and pick the largest (assumed to be the hand)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ''  # nothing detected
    
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    if w < 10 or h < 10:
        return ''  # too small to be valid
    
    # 4. Extract ROI, resize to 28×28 and flatten
    roi = thresh[y:y+h, x:x+w]
    roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi_flat = roi_resized.flatten().reshape(1, -1).astype(np.float32) / 255.0
    
    # 5. Predict with your model
    label = model.predict(roi_flat)[0]
    return str(label)
