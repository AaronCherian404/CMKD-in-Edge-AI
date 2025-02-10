from service import FogService
from flask import Flask, request, jsonify

app = Flask(__name__)
fog_service = FogService()

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    result = fog_service.process_data(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)