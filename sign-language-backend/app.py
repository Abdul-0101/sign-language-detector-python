from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your trained model
with open("model.p", "rb") as f:
    model_dict = pickle.load(f)
model = model_dict["model"]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    landmarks = data.get("landmarks")
    if not landmarks or len(landmarks) != 42:
        return jsonify({"error": "Invalid input"}), 400
    try:
        prediction = model.predict([np.asarray(landmarks)])
        return jsonify({"letter": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
