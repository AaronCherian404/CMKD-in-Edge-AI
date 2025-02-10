from inference import EdgeInference
from flask import Flask, request, jsonify

app = Flask(__name__)
edge_inference = EdgeInference()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    prediction, probabilities = edge_inference.predict(data)
    return jsonify({
        'prediction': prediction,
        'probabilities': probabilities
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002)