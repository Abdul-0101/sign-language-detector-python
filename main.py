from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("✅ Received POST:", data)
        return jsonify({
            "letter": "A",
            "confirmed": True,
            "current_text": "A",
            "predicted_word": "ARE"
        })
    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": "Failed", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
