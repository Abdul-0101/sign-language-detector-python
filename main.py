from flask import Flask, render_template, Response, redirect, url_for
import cv2

# import everything your inference module exposes:
from inference_classifier import (
    predict,
    update_current_word,
    get_current_word,
    reset_current_word,
    speak
)

app = Flask(__name__)
camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset')
def reset():
    reset_current_word()
    return redirect(url_for('index'))

@app.route('/inference-code')
def show_inference_code():
    with open('inference_classifier.py', 'r') as f:
        code = f.read()
    return render_template('inference.html', code=code)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # 1) run your full inference pipeline
        letter, confidence, frame = predict(frame)

        # 2) draw ROI
        cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)

        # 3) only append & speak if confident
        if confidence > 0.7:
            update_current_word(letter)
            speak(get_current_word())   # will regenerate static/tts.mp3
            cv2.putText(
                frame,
                f"{letter} ({confidence*100:.1f}%)",
                (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2
            )

        # 4) overlay the cumulatively formed word
        word = get_current_word()
        cv2.putText(
            frame,
            f"Word: {word}",
            (100, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2
        )

        # 5) stream it
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        )

@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
