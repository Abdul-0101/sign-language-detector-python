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

# 3. Load dictionary (for word prediction if needed)
with open('dictionary.txt', 'r') as f:
    DICTIONARY = set(line.strip().upper() for line in f)

# 4. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, min_detection_confidence=0.9)

# 5. Define homepage route
@app.route('/')
def index():
    return render_template('index.html')  # Front-end for webcam and predictions

# 6. Define prediction API route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return jsonify(letter="")

    # Extract hand landmarks
    hand_landmarks = result.multi_hand_landmarks[0]
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    feature_vector = []

    for lm in hand_landmarks.landmark:
        feature_vector.append(lm.x - min(xs))
        feature_vector.append(lm.y - min(ys))

    feature_vector = np.asarray(feature_vector).reshape(1, -1)
    prediction = model.predict(feature_vector)[0]

    return jsonify(letter=prediction)

# 7. Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Very important for Render!
    app.run(host='0.0.0.0', port=port)
