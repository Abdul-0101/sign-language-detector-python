# main.py
from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
import pickle

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

# Load your trained model once
with open('model.p', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    """Serve the main page with browser-based webcam capture."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    """
    Receive a base64-encoded JPEG frame from the client,
    run your Python classifier on it, and return the label.
    """
    data = request.get_json(force=True)
    img_b64 = data.get('image', '').split(',', 1)[1]
    img_bytes = base64.b64decode(img_b64)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, flags=cv2.IMREAD_COLOR)

    # Your inference logic, identical to inference_classifier.py
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return jsonify(label="")
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    if w < 10 or h < 10:
        return jsonify(label="")
    roi = thresh[y:y+h, x:x+w]
    roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi_flat = roi_resized.flatten().reshape(1, -1).astype(np.float32) / 255.0
    label = model.predict(roi_flat)[0]

    return jsonify(label=str(label))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
