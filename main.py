from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import cv2
import mediapipe as mp
from gtts import gTTS
import io
import base64
import os

app = Flask(__name__)

# Load model
with open("model.p", "rb") as f:
    model = pickle.load(f)['model']

# Load dictionary
with open("dictionary.txt", "r") as f:
    DICTIONARY = [line.strip().upper() for line in f]

# MediaPipe hands
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, min_detection_confidence=0.9)
mp_draw = mp.solutions.drawing_utils

# State
paragraph = ""
last_letter = ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global paragraph, last_letter

    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return jsonify(letter="", paragraph=paragraph)

    hand = result.multi_hand_landmarks[0]
    xs = [lm.x for lm in hand.landmark]
    ys = [lm.y for lm in hand.landmark]
    feat = [(lm.x - min(xs), lm.y - min(ys)) for lm in hand.landmark]
    flat_feat = np.concatenate(feat)

    pred = model.predict([flat_feat])[0]

    if pred != last_letter:
        paragraph += pred
        last_letter = pred

    return jsonify(letter=pred, paragraph=paragraph)

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
    paragraph += " "
    return jsonify(paragraph=paragraph)

@app.route('/newline', methods=['POST'])
def newline():
    global paragraph
    paragraph += "\n"
    return jsonify(paragraph=paragraph)

@app.route('/clear', methods=['POST'])
def clear():
    global paragraph
    paragraph = ""
    return jsonify(paragraph=paragraph)

@app.route('/correction', methods=['POST'])
def correction():
    global paragraph
    paragraph = request.form.get('corrected', '')
    return jsonify(paragraph=paragraph)

@app.route('/speak', methods=['GET'])
def speak():
    tts = gTTS(text=paragraph, lang='en')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    audio_base64 = base64.b64encode(mp3_fp.read()).decode('utf-8')
    return jsonify(audio=audio_base64)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
