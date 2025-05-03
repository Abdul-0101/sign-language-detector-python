import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# Load the dictionary
DICTIONARY_FILE = "dictionary.txt"

def load_dictionary():
    if os.path.exists(DICTIONARY_FILE):
        with open(DICTIONARY_FILE, "r") as file:
            return set(word.strip().upper() for word in file.readlines())
    return set()

dictionary = load_dictionary()

def predict_word(prefix):
    if not prefix:
        return ""
    prefix = prefix.upper()
    matches = [word for word in dictionary if word.startswith(prefix)]
    if matches:
        return max(matches, key=len)
    return ""

# Load the trained model
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.9)

def extract_landmark_features(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    lms = res.multi_hand_landmarks[0].landmark
    xs, ys = [lm.x for lm in lms], [lm.y for lm in lms]
    area = (max(xs)-min(xs))*(max(ys)-min(ys))
    if not (0.02 <= area <= 0.2):
        return None
    feats = []
    for lm in lms:
        feats += [lm.x - min(xs), lm.y - min(ys)]
    return np.array(feats).reshape(1, -1)

def classify_frame(frame):
    feats = extract_landmark_features(frame)
    if feats is None:
        return {"letter": "", "word": ""}
    letter = model.predict(feats)[0]
    word = predict_word(letter)
    return {"letter": letter, "word": word}
