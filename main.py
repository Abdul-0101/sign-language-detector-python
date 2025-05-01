from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from gtts import gTTS
import base64
import io
import os

app = Flask(__name__)

# Load pre-trained model (unchanged)
with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

# Load dictionary for suggestions
with open('dictionary.txt', 'r') as f:
    dictionary = set(w.strip().upper() for w in f)

def predict_word(prefix):
    prefix = prefix.upper()
    matches = [w for w in dictionary if w.startswith(prefix)]
    return max(matches, key=len) if matches else ''

# Application state
paragraph = ''
current_text = ''
last_letter = None
stable_count = 0
STABLE_THRESHOLD = 5
waiting_removal = False
absent_count = 0
ABSENT_THRESHOLD = 2

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global paragraph, current_text, last_letter, stable_count, waiting_removal, absent_count

    data = request.get_json()
    features = data.get('features', [])
    hand_present = data.get('hand_present', False)

    # Wait for hand removal to start next letter
    if waiting_removal:
        if not hand_present:
            absent_count += 1
            if absent_count >= ABSENT_THRESHOLD:
                waiting_removal = False
                absent_count = 0
                last_letter = None
        else:
            absent_count = 0
        return jsonify(letter='', current=current_text, paragraph=paragraph)

    # Detect letter when hand is present
    if hand_present and len(features) == 42:
        idx = model.predict([features])[0]
        if idx == last_letter:
            stable_count += 1
        else:
            last_letter = idx
            stable_count = 1

        if stable_count >= STABLE_THRESHOLD:
            letter = chr(65 + int(idx))
            current_text += letter
            stable_count = 0
            waiting_removal = True
            return jsonify(letter=letter, current=current_text, paragraph=paragraph)

    return jsonify(letter='', current=current_text, paragraph=paragraph)

@app.route('/backspace', methods=['POST'])
def backspace():
    global current_text, paragraph
    if current_text:
        current_text = current_text[:-1]
    elif paragraph:
        paragraph = paragraph[:-1]
    return jsonify(current=current_text, paragraph=paragraph)

@app.route('/space', methods=['POST'])
def space():
    global current_text, paragraph
    if current_text:
        paragraph += current_text + ' '
        current_text = ''
    return jsonify(current=current_text, paragraph=paragraph)

@app.route('/newline', methods=['POST'])
def newline():
    global current_text, paragraph
    if current_text:
        paragraph += current_text + '\n'
        current_text = ''
    else:
        paragraph += '\n'
    return jsonify(current=current_text, paragraph=paragraph)

@app.route('/clear', methods=['POST'])
def clear():
    global paragraph, current_text
    paragraph = ''
    current_text = ''
    return jsonify(current=current_text, paragraph=paragraph)

@app.route('/correction', methods=['POST'])
def correction():
    global current_text, dictionary
    corr = request.form.get('corrected', '').upper()
    if corr:
        current_text = corr
        dictionary.add(corr)
    return jsonify(current=current_text, paragraph=paragraph)

@app.route('/speak', methods=['GET'])
def speak():
    text = paragraph if paragraph else current_text
    if not text:
        return jsonify(audio='')
    tts = gTTS(text=text, lang='en')
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return jsonify(audio=base64.b64encode(buf.read()).decode())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
