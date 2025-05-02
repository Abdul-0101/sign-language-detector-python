from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and dictionary
with open('model.p', 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']

# Load dictionary
with open('dictionary.txt') as f:
    DICTIONARY = set(line.strip().upper() for line in f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('landmarks', [])
    # data is flat list: [x0, y0, x1, y1, ..., x20, y20]
    arr = np.array(data).reshape(1, -1)
    letter = model.predict(arr)[0]
    return jsonify({'letter': letter})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
