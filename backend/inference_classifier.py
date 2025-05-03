import cv2, mediapipe as mp, numpy as np, pickle, os

# — Dictionary persistence —
DICTIONARY_FILE = "dictionary.txt"
def load_dictionary():
    if os.path.exists(DICTIONARY_FILE):
        return set(w.strip().upper() for w in open(DICTIONARY_FILE))
    return set()
dictionary = load_dictionary()

def predict_word(prefix):
    matches = [w for w in dictionary if w.startswith(prefix.upper())]
    return max(matches, key=len) if matches else ""

# — Load trained RF model —
with open("model.p","rb") as f:
    model = pickle.load(f)["model"]

# — MediaPipe hands setup & feature extraction —
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
        feats += [lm.x-min(xs), lm.y-min(ys)]
    return np.array(feats).reshape(1, -1)

def classify_frame(frame):
    feats = extract_landmark_features(frame)
    if feats is None:
        return {"letter":"", "word":""}
    letter = model.predict(feats)[0]
    word = predict_word(letter)
    return {"letter": letter, "word": word}
