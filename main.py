from flask import Flask, render_template, Response, redirect, url_for
import cv2
from inference_classifier import predict, update_current_word, get_current_word, reset_current_word

app = Flask(__name__)
camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    # Main page: live feed + reset link + view source link
    return render_template('index.html')

@app.route('/reset')
def reset():
    # Clear the current word buffer
    reset_current_word()
    return redirect(url_for('index'))

@app.route('/inference-code')
def show_inference_code():
    # Serve the full inference_classifier.py source as plain text
    with open('inference_classifier.py', 'r') as f:
        code = f.read()
    return render_template('inference.html', code=code)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Run your full inference logic
        letter, confidence, frame = predict(frame)

        # Draw ROI
        cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)

        # Update and display letter if confident
        if confidence > 0.7:
            update_current_word(letter)
            cv2.putText(
                frame,
                f"{letter} ({confidence*100:.1f}%)",
                (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2
            )

        # Overlay the cumulatively formed word
        word = get_current_word()
        cv2.putText(
            frame,
            f"Word: {word}",
            (100, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2
        )

        # JPEG-encode and stream
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

@app.route('/video')
def video():
    # Video streaming route. Put this in your <img src="/video">
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    # debug=True for development; turn off in production
    app.run(host='0.0.0.0', port=5000, debug=True)
