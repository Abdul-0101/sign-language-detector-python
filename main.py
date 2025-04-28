from flask import Flask, render_template, request, jsonify
import pickle
import os
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import io
import base64

app = Flask(__name__)

# Load the model
with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

# Load dictionary
with open('dictionary.txt', 'r') as f:
    DICTIONARY = [line.strip().upper() for line in f]

mp_hands = mp.solutions.hands.Hands(static_image_mode=False, min_detection_confidence=0.9)

# Variables
stable_detection_threshold = 3
stable_count = 0
last_detected_letter = ""
current_confirmed_letter = ""
paragraph = ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global stable_count, last_detected_letter, current_confirmed_letter, paragraph

    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        stable_count = 0
        last_detected_letter = ""
        return jsonify(letter="", paragraph=paragraph)

    hand_landmarks = result.multi_hand_landmarks[0]
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    feature_vector = []

    for lm in hand_landmarks.landmark:
        feature_vector.append(lm.x - min(xs))
        feature_vector.append(lm.y - min(ys))

    feature_vector = np.asarray(feature_vector).reshape(1, -1)
    prediction = model.predict(feature_vector)[0]

    if prediction == last_detected_letter:
        stable_count += 1
    else:
        stable_count = 1
        last_detected_letter = prediction

    if stable_count >= stable_detection_threshold:
        current_confirmed_letter = prediction
        paragraph += prediction
        stable_count = 0

    return jsonify(letter=current_confirmed_letter, paragraph=paragraph)

@app.route('/predict_word', methods=['GET'])
def predict_word():
    partial = request.args.get('partial', '').upper()
    matches = [w for w in DICTIONARY if w.startswith(partial)]
    return jsonify(predictions=matches[:3])

@app.route('/backspace', methods=['POST'])
def backspace():
    global paragraph
    paragraph = paragraph[:-1]
    return jsonify(paragraph=paragraph)

@app.route('/space', methods=['POST'])
def space():
    global paragraph
    paragraph += ' '
    return speak()

@app.route('/newline', methods=['POST'])
def newline():
    global paragraph
    paragraph += '\n'
    return speak()

@app.route('/clear', methods=['POST'])
def clear():
    global paragraph
    paragraph = ""
    return jsonify(paragraph=paragraph)

@app.route('/correction', methods=['POST'])
def correction():
    global paragraph
    corrected = request.form.get('corrected', '')
    paragraph = corrected
    return jsonify(paragraph=paragraph)

@app.route('/speak', methods=['GET'])
def speak():
    tts = gTTS(text=paragraph, lang='en')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    audio_base64 = base64.b64encode(mp3_fp.read()).decode('utf-8')
    return jsonify(audio=audio_base64, paragraph=paragraph)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
