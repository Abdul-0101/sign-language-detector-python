# main.py

from flask import Flask, render_template, request, jsonify
import pickle
import os
import cv2
import mediapipe as mp
import numpy as np

# 1. Create Flask app
app = Flask(__name__)

# 2. Load trained model
with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

# 3. Load dictionary (optional for word prediction later)
with open('dictionary.txt', 'r') as f:
    DICTIONARY = set(line.strip().upper() for line in f)

# 4. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, min_detection_confidence=0.9)

# 5. Stability Control Variables
stable_detection_threshold = 3  # How many frames before confirming a letter
stable_count = 0
last_detected_letter = ""
current_confirmed_letter = ""

# 6. Home page
@app.route('/')
def index():
    return render_template('index.html')

# 7. Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    global stable_count, last_detected_letter, current_confirmed_letter

    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    img = cv2.flip(img, 1)  # 🔥 This correctly flips the camera horizontally

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        # No hand detected: reset stability counters
        stable_count = 0
        last_detected_letter = ""
        return jsonify(letter="")

    # Extract landmarks
    hand_landmarks = result.multi_hand_landmarks[0]
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    feature_vector = []

    for lm in hand_landmarks.landmark:
        feature_vector.append(lm.x - min(xs))
        feature_vector.append(lm.y - min(ys))

    feature_vector = np.asarray(feature_vector).reshape(1, -1)
    prediction = model.predict(feature_vector)[0]

    # Stability logic
    if prediction == last_detected_letter:
        stable_count += 1
    else:
        stable_count = 1
        last_detected_letter = prediction

    if stable_count >= stable_detection_threshold:
        current_confirmed_letter = prediction
    else:
        current_confirmed_letter = ""

    return jsonify(letter=current_confirmed_letter)

# 8. Start the server
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Required for Render deployment
    app.run(host='0.0.0.0', port=port)
