# inference_classifier.py
import cv2, mediapipe as mp, numpy as np, pickle, os

# --- Load model ---
with open("model.p", "rb") as f:
    model = pickle.load(f)["model"]

# --- Dictionary Prediction ---
DICTIONARY_FILE = "dictionary.txt"
def load_dictionary():
    if os.path.exists(DICTIONARY_FILE):
        return set(w.strip().upper() for w in open(DICTIONARY_FILE))
    return set()

dictionary = load_dictionary()

def predict_word(prefix):
    matches = [w for w in dictionary if w.startswith(prefix.upper())]
    return max(matches, key=len) if matches else ""

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.9)

# --- Core Inference Logic ---
def extract_landmark_features(frame):
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None
    lms = results.multi_hand_landmarks[0].landmark
    xs = [lm.x for lm in lms]
    ys = [lm.y for lm in lms]
    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    if not (0.02 <= area <= 0.2):
        return None
    feats = []
    for lm in lms:
        feats += [lm.x - min(xs), lm.y - min(ys)]
    return np.array(feats).reshape(1, -1)

last_detected_letter = ""
stable_count = 0
stable_threshold = 3
waiting_for_removal = False
hand_absent_count = 0
hand_absent_threshold = 2
current_text = ""

def classify_frame(frame):
    global last_detected_letter, stable_count, waiting_for_removal
    global hand_absent_count, current_text

    feats = extract_landmark_features(frame)
    if feats is None:
        hand_absent_count += 1
        if waiting_for_removal and hand_absent_count >= hand_absent_threshold:
            waiting_for_removal = False
            last_detected_letter = ""
        return {"letter": "", "word": predict_word(current_text)}

    hand_absent_count = 0
    if waiting_for_removal:
        return {"letter": "", "word": predict_word(current_text)}

    detected_letter = model.predict(feats)[0]
    if detected_letter == last_detected_letter:
        stable_count += 1
    else:
        last_detected_letter = detected_letter
        stable_count = 1

    if stable_count >= stable_threshold:
        current_text += detected_letter
        waiting_for_removal = True
        stable_count = 0

    return {"letter": detected_letter, "word": predict_word(current_text)}
