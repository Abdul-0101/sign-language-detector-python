from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        landmarks = data.get("landmarks", [])

        print("📥 Received landmarks:", landmarks)

        if len(landmarks) != 42:
            return jsonify({"error": "Invalid number of landmarks"}), 400

        # Dummy prediction — always return "A"
        letter = "A"

        return jsonify({
            "letter": letter,
            "confirmed": True,
            "current_text": letter,
            "predicted_word": "A"
        })

    except Exception as e:
        return jsonify({"error": "Server error", "message": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    return jsonify({"status": "reset"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
