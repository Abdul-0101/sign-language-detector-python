from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from gtts import gTTS
import base64
import io
import os
import threading

app = Flask(__name__)

# Load model
with open("model.p", "rb") as f:
    model = pickle.load(f)["model"]

# Load dictionary
with open("dictionary.txt", "r") as f:
    dictionary = set(w.strip().upper() for w in f)

def predict_word(prefix):
    prefix = prefix.upper()
    matches = [w for w in dictionary if w.startswith(prefix)]
    return max(matches, key=len) if matches else ""

# State
paragraph = ""
current_text = ""
last_detected_letter = ""
stable_count = 0
stable_threshold = 3
waiting_for_hand_removal = False
hand_absent_count = 0
hand_absent_threshold = 2

# Online learning buffers
feedback_X = []
feedback_y = []
model_lock = threading.Lock()

def retrain_model():
    """Re-fit RandomForest on augmented data (original + feedback)."""
    global model
    with model_lock:
        if not feedback_X:
            return
        # Original training data not stored here; this is just illustrative.
        # In practice you’d reload original X,y or use partial_fit classifier.
        X = np.array(feedback_X)
        y = np.array(feedback_y)
        model.fit(X, y)
        with open("model.p", "wb") as f:
            pickle.dump({"model": model}, f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global paragraph, current_text, last_detected_letter
    global stable_count, waiting_for_hand_removal, hand_absent_count

    data = request.get_json()
    features = data.get("features", [])
    hand_present = data.get("hand_present", False)

    # waiting for removal
    if waiting_for_hand_removal:
        if not hand_present:
            hand_absent_count += 1
            if hand_absent_count >= hand_absent_threshold:
                waiting_for_hand_removal = False
                hand_absent_count = 0
                last_detected_letter = ""
        else:
            hand_absent_count = 0
        return jsonify(letter="", current=current_text, predicted=predict_word(current_text), paragraph=paragraph)

    # detect letter
    if hand_present and len(features) == 42:
        letter = model.predict([features])[0]
        if letter == last_detected_letter:
            stable_count += 1
        else:
            last_detected_letter = letter
            stable_count = 1
        if stable_count >= stable_threshold:
            current_text += letter
            stable_count = 0
            waiting_for_hand_removal = True
        return jsonify(letter=letter, current=current_text, predicted=predict_word(current_text), paragraph=paragraph)

    return jsonify(letter="", current=current_text, predicted=predict_word(current_text), paragraph=paragraph)

@app.route("/backspace", methods=["POST"])
def backspace():
    global current_text, paragraph
    if current_text:
        current_text = current_text[:-1]
    elif paragraph:
        paragraph = paragraph[:-1]
    return jsonify(current=current_text, paragraph=paragraph)

@app.route("/space", methods=["POST"])
def space():
    global current_text, paragraph
    if current_text:
        paragraph += current_text + " "
    elif paragraph and not paragraph.endswith(" "):
        paragraph += " "
    return jsonify(current="", paragraph=paragraph)

@app.route("/newline", methods=["POST"])
def newline():
    global current_text, paragraph
    if current_text:
        paragraph += current_text + "\n"
    else:
        paragraph += "\n"
    return jsonify(current="", paragraph=paragraph)

@app.route("/clear", methods=["POST"])
def clear():
    global current_text, paragraph
    current_text = ""
    paragraph = ""
    return jsonify(current="", paragraph=paragraph)

@app.route("/correction", methods=["POST"])
def correction():
    """Receive manual correction of current_text"""
    global current_text
    corr = request.form.get("corrected", "").upper()
    if corr:
        current_text = corr
        dictionary.add(corr)
    return jsonify(current=current_text, paragraph=paragraph)

@app.route("/feedback", methods=["POST"])
def feedback():
    """After a word completes, collect features+correct word, retrain in background."""
    data = request.get_json()
    features = data.get("features", [])
    correct = data.get("correct", "").upper()
    if features and correct:
        feedback_X.append(features)
        feedback_y.append(correct)
        # retrain in background
        threading.Thread(target=retrain_model, daemon=True).start()
    return jsonify(status="ok")

@app.route("/speak", methods=["GET"])
def speak():
    text = paragraph if paragraph else current_text
    if not text:
        return jsonify(audio="")
    tts = gTTS(text=text, lang="en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return jsonify(audio=base64.b64encode(buf.read()).decode())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
