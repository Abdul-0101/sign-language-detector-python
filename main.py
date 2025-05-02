from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    landmarks = data.get("landmarks", [])
    if len(landmarks) != 42:
        return jsonify({"error": "Invalid landmark data"}), 400

    # Dummy prediction
    return jsonify({
        "letter": "A",
        "confirmed": True,
        "current_text": "A",
        "predicted_word": "APPLE"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)