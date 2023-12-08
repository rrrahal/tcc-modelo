# app.py
from flask import Flask, request, jsonify
from ml_functions import load_model, evaluate
from flask_cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = load_model()

@app.route('/heartbeat')
def heartbeat():
    result = evaluate('blabla', model)
    print(result)
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        data = request.get_json()
        description = data['description']

        result = evaluate(description, model)

        output = { "is_green": result }

        return jsonify(output)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5010)
