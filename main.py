# ✅ Final Plan: Merge Local `inference_classifier.py` Logic into Flask App

# main.py (Updated with all local logic and features, gTTS only, no voice input)

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from gtts import gTTS
import base64
import io
import os

app = Flask(__name__)

with open("model.p", "rb") as f:
    model = pickle.load(f)["model"]

with open("dictionary.txt", "r") as f:
    dictionary = set(word.strip().upper() for word in f)

def predict_word(prefix):
    if not prefix:
        return ""
    prefix = prefix.upper()
    matches = [word for word in dictionary if word.startswith(prefix)]
    if matches:
        return max(matches, key=len)
    return ""

paragraph = ""
current_text = ""
waiting_for_hand_removal = False
hand_absent_count = 0
hand_absent_threshold = 2
last_detected_letter = ""
stable_count = 0
stable_threshold = 3

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global paragraph, current_text, last_detected_letter, stable_count, waiting_for_hand_removal, hand_absent_count

    data = request.get_json()
    features = data.get("features", [])
    hand_present = data.get("hand_present", False)

    if waiting_for_hand_removal:
        if not hand_present:
            hand_absent_count += 1
            if hand_absent_count >= hand_absent_threshold:
                waiting_for_hand_removal = False
                hand_absent_count = 0
                last_detected_letter = ""
        else:
            hand_absent_count = 0
        return jsonify(letter="", current=current_text, paragraph=paragraph)

    if hand_present and len(features) == 42:
        detected_letter = model.predict([features])[0]
        if detected_letter == last_detected_letter:
            stable_count += 1
        else:
            last_detected_letter = detected_letter
            stable_count = 1
        if stable_count >= stable_threshold:
            current_text += detected_letter
            stable_count = 0
            waiting_for_hand_removal = True
        predicted = predict_word(current_text)
        return jsonify(letter=detected_letter, current=current_text, predicted=predicted, paragraph=paragraph)

    return jsonify(letter="", current=current_text, paragraph=paragraph)

@app.route("/backspace", methods=["POST"])
def backspace():
    global current_text, paragraph
    if current_text:
        current_text = current_text[:-1]
    elif paragraph:
        paragraph = paragraph[:-1]
    return jsonify(paragraph=paragraph, current=current_text)

@app.route("/space", methods=["POST"])
def space():
    global current_text, paragraph
    if current_text:
        paragraph += current_text + " "
        current_text = ""
    elif paragraph and not paragraph.endswith(" "):
        paragraph += " "
    return jsonify(paragraph=paragraph, current=current_text)

@app.route("/newline", methods=["POST"])
def newline():
    global current_text, paragraph
    if current_text:
        paragraph += current_text + "\n"
        current_text = ""
    else:
        paragraph += "\n"
    return jsonify(paragraph=paragraph, current=current_text)

@app.route("/clear", methods=["POST"])
def clear():
    global paragraph, current_text
    paragraph = ""
    current_text = ""
    return jsonify(paragraph="", current="")

@app.route("/correction", methods=["POST"])
def correction():
    global paragraph, current_text
    correction = request.form.get("corrected", "")
    if correction:
        dictionary.add(correction.upper())
        current_text = correction
    return jsonify(paragraph=paragraph, current=current_text)

@app.route("/speak", methods=["GET"])
def speak():
    text = paragraph if paragraph else current_text
    if not text:
        return jsonify(audio="")
    tts = gTTS(text=text, lang="en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    return jsonify(audio=encoded)

if __name__ == "__main__":
    app.run(debug=True)
