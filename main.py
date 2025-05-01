from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from gtts import gTTS
import base64
import io
import os
from sklearn.linear_model import SGDClassifier

app = Flask(__name__)

# Load or initialize an incremental model
MODEL_PATH = "model.p"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)["model"]
else:
    # Create initial SGD classifier for 26 letters (0=A,...)
    model = SGDClassifier(loss="log", max_iter=1000, tol=1e-3)
    # Dummy partial_fit with all classes to initialize
    model.partial_fit(np.zeros((26, 42)), list(range(26)), classes=list(range(26)))
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model}, f)

# Load dictionary for word prediction
with open("dictionary.txt", "r") as f:
    dictionary = set(w.strip().upper() for w in f)

def predict_word(prefix):
    prefix = prefix.upper()
    matches = [w for w in dictionary if w.startswith(prefix)]
    return max(matches, key=len) if matches else ""

# App state
paragraph = ""
current_text = ""
last_letter = ""
stable_count = 0
STABLE_THRESHOLD = 3
waiting_removal = False
absent_count = 0
ABSENT_THRESHOLD = 2

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"] )
def predict():
    global paragraph, current_text, last_letter, stable_count, waiting_removal, absent_count
    data = request.get_json()
    features = data.get("features", [])
    hand_present = data.get("hand_present", False)
    
    # If waiting for user to remove hand
    if waiting_removal:
        if not hand_present:
            absent_count += 1
            if absent_count >= ABSENT_THRESHOLD:
                waiting_removal = False
                absent_count = 0
                last_letter = ""
        else:
            absent_count = 0
        return jsonify(letter="", current=current_text, predicted=predict_word(current_text), paragraph=paragraph)
    
    # If hand present and valid features
    if hand_present and len(features) == 42:
        letter_idx = model.predict([features])[0]
        # Map idx to letter
        letter = chr(ord('A') + int(letter_idx))
        # stability
        if letter == last_letter:
            stable_count += 1
        else:
            last_letter = letter
            stable_count = 1
        if stable_count >= STABLE_THRESHOLD:
            current_text += letter
            stable_count = 0
            waiting_removal = True
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
        # live feedback prompt via front-end
        current_text = ""
    elif paragraph and not paragraph.endswith(" "):
        paragraph += " "
    return jsonify(current=current_text, paragraph=paragraph)

@app.route("/newline", methods=["POST"])
def newline():
    global current_text, paragraph
    if current_text:
        paragraph += current_text + "\n"
        current_text = ""
    else:
        paragraph += "\n"
    return jsonify(current=current_text, paragraph=paragraph)

@app.route("/clear", methods=["POST"])
def clear():
    global paragraph, current_text
    paragraph, current_text = "", ""
    return jsonify(current="", paragraph="")

@app.route("/correction", methods=["POST"])
def correction():
    global current_text, dictionary
    corr = request.form.get("corrected", "").upper()
    if corr:
        current_text = corr
        dictionary.add(corr)
    return jsonify(current=current_text, paragraph=paragraph)

@app.route("/feedback", methods=["POST"])
def feedback():
    global model
    data = request.get_json()
    features = data.get("features", [])
    correct = data.get("correct", "").upper()
    if len(features) == 42 and correct:
        y = ord(correct[0]) - 65
        # online update
        model.partial_fit([features], [y])
        # save model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": model}, f)
    return jsonify(status="ok")

@app.route("/speak", methods=["GET"])
def speak():
    text = paragraph or current_text
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
