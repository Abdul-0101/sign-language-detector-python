from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
import inference_classifier  # the file you just updated

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json(force=True)
    # strip off "data:image/jpeg;base64,"
    img_b64 = data['image'].split(',', 1)[1]
    img_bytes = base64.b64decode(img_b64)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Delegate to your inference module
    label = inference_classifier.predict(frame)
    return jsonify(label=label)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
