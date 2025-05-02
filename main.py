from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os

# Initialize Flask
app = Flask(__name__)

# Load model
with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

# Load dictionary
DICTIONARY_FILE = "dictionary.txt"
if os.path.exists(DICTIONARY_FILE):
    with open(DICTIONARY_FILE, "r") as file:
        dictionary = set(word.strip().upper() for word in file.readlines())
else:
    dictionary = set()

# State variables
stable_detection_threshold = 3
stable_count = 0
last_detected_letter = ""
current_text = ""

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global stable_count, last_detected_letter, current_text

    data = request.get_json()
    landmarks = data.get("landmarks")  # Expecting 42 values (21 x/y pairs)

    if not landmarks or len(landmarks) != 42:
        return jsonify({"error": "Invalid landmark data"}), 400

    prediction = model.predict([np.array(landmarks)])[0]

    if prediction == last_detected_letter:
        stable_count += 1
    else:
        stable_count = 1
        last_detected_letter = prediction

    confirmed = False
    if stable_count >= stable_detection_threshold:
        current_text += prediction
        stable_count = 0
        confirmed = True

    predicted_word = ""
    prefix = current_text.upper()
    if prefix:
        matches = [word for word in dictionary if word.startswith(prefix)]
        if matches:
            predicted_word = max(matches, key=len)

    return jsonify({
        "letter": prediction,
        "confirmed": confirmed,
        "current_text": current_text,
        "predicted_word": predicted_word
    })

@app.route('/reset', methods=['POST'])
def reset():
    global current_text, last_detected_letter, stable_count
    current_text = ""
    last_detected_letter = ""
    stable_count = 0
    return jsonify({"status": "reset"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
