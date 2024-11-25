from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)


model = pipeline("sentiment-analysis")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    result = model(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
