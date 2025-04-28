from flask import Flask, render_template, request, jsonify, send_file
import pickle
import os
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import io

app = Flask(__name__)

# Load model
with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

# Load dictionary
with open('dictionary.txt', 'r') as f:
    DICTIONARY = set(line.strip().upper() for line in f)

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, min_detection_confidence=0.9)

# Stability Control Variables
stable_detection_threshold = 3
stable_count = 0
last_detected_letter = ""
current_confirmed_letter = ""
forming_word = ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global stable_count, last_detected_letter, current_confirmed_letter, forming_word

    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.flip(img, 1)  # Correct flip

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        stable_count = 0
        last_detected_letter = ""
        return jsonify(letter="", word=forming_word)

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
        forming_word += prediction
        stable_count = 0

    return jsonify(letter=current_confirmed_letter, word=forming_word)

@app.route('/backspace', methods=['POST'])
def backspace():
    global forming_word
    forming_word = forming_word[:-1]
    return jsonify(word=forming_word)

@app.route('/space', methods=['POST'])
def space():
    global forming_word
    forming_word += ' '
    return jsonify(word=forming_word)

@app.route('/newline', methods=['POST'])
def newline():
    global forming_word
    forming_word += '\n'
    return jsonify(word=forming_word)

@app.route('/correction', methods=['POST'])
def correction():
    global forming_word
    corrected = request.form.get('corrected', '')
    forming_word = corrected
    return jsonify(word=forming_word)

@app.route('/speak', methods=['GET'])
def speak():
    tts = gTTS(text=forming_word, lang='en')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return send_file(mp3_fp, mimetype="audio/mpeg")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
