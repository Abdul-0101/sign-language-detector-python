from flask import Flask, request, jsonify, render_template
import pickle, numpy as np, os, base64, io
from gtts import gTTS
from sklearn.linear_model import SGDClassifier

app = Flask(__name__)

# Load or initialize incremental model
MODEL_PATH = 'model.p'
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)['model']
else:
    model = SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
    # initialize classes A–Z
    dummy_X = np.zeros((26, 42))
    dummy_y = np.arange(26)
    model.partial_fit(dummy_X, dummy_y, classes=dummy_y)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model}, f)

# Load dictionary
with open('dictionary.txt', 'r') as f:
    dictionary = set(w.strip().upper() for w in f)

def predict_word(prefix):
    prefix = prefix.upper()
    matches = [w for w in dictionary if w.startswith(prefix)]
    return max(matches, key=len) if matches else ''

# State variables
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

    # wait for hand removal to start next letter
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

    # detect and confirm letter
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

@app.route('/space', methods=['POST'])
def space():
    global paragraph, current_text
    if current_text:
        paragraph += current_text + ' '
        current_text = ''
    return jsonify(current=current_text, paragraph=paragraph)

@app.route('/newline', methods=['POST'])
def newline():
    global paragraph, current_text
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
