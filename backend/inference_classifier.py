import cv2, mediapipe as mp, numpy as np, pickle, os
from collections import deque, Counter

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

# — MediaPipe hands setup —
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.9)

# — State for stability & text accumulation —
prediction_buffer = deque(maxlen=5)
stable_threshold = 3
current_text = ""
waiting_for_removal = False
hand_absent_count = 0
hand_absent_threshold = 2

def classify_frame(frame):
    global current_text, waiting_for_removal, hand_absent_count

    # flip to match training
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    # no hand detected
    if not res.multi_hand_landmarks:
        hand_absent_count += 1
        if waiting_for_removal and hand_absent_count >= hand_absent_threshold:
            waiting_for_removal = False
            prediction_buffer.clear()
        return {"letter":"", "word": predict_word(current_text)}

    hand_absent_count = 0
    if waiting_for_removal:
        # still waiting for hand removal
        return {"letter":"", "word": predict_word(current_text)}

    # extract landmarks
    lms = res.multi_hand_landmarks[0].landmark
    xs = [lm.x for lm in lms]; ys = [lm.y for lm in lms]
    area = (max(xs)-min(xs))*(max(ys)-min(ys))
    # tighten area thresholds
    if not (0.03 <= area <= 0.12):
        return {"letter":"", "word": predict_word(current_text)}

    # build feature vector
    feats = []
    min_x, min_y = min(xs), min(ys)
    for lm in lms:
        feats += [lm.x-min_x, lm.y-min_y]

    # predict letter
    letter = model.predict([np.asarray(feats)])[0]
    prediction_buffer.append(letter)

    # majority vote
    most_common, count = Counter(prediction_buffer).most_common(1)[0]
    if count >= stable_threshold:
        # commit letter
        current_text += most_common
        waiting_for_removal = True
        prediction_buffer.clear()
        return {"letter": most_common, "word": predict_word(current_text)}

    return {"letter":"", "word": predict_word(current_text)}
