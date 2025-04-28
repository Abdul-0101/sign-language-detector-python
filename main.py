from flask import Flask, render_template, Response, request, jsonify
import cv2
import base64
import numpy as np
import inference_classifier  # your existing Python inference module

app = Flask(__name__,
            static_folder='static',        # serve JS, dictionary.txt here
            template_folder='templates')   # your index.html lives here

# Open the default camera
cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

def gen_frames():
    """Generator that yields camera frames as multipart MJPEG."""
    while True:
        success, frame = cap.read()
        if not success:
            break
        # encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in an <img> src."""
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/predict', methods=['POST'])
def predict_route():
    """
    Receive a base64-encoded JPEG frame from the client,
    run your Python classifier on it, and return the label.
    """
    data = request.get_json(force=True)
    # data['image'] is like "data:image/jpeg;base64,/9j/4AAQ..."
    img_b64 = data.get('image', '').split(',', 1)[1]
    img_bytes = base64.b64decode(img_b64)
    # convert bytes to OpenCV image
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, flags=cv2.IMREAD_COLOR)
    # call your existing classifier
    label = inference_classifier.predict(frame)
    return jsonify(label=label)

if __name__ == '__main__':
    # start Flask app
    app.run(host='0.0.0.0', port=5000, threaded=True)
