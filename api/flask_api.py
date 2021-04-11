from flask import Flask, Response, jsonify, request, redirect
from flask_cors import CORS
import pandas as pd
import json
import sys

sys.path.append('..')
import models.train as models

app = Flask(__name__)
cors = CORS(app)

# ML model definition
loaded_models = {
    'plain-1': 'data/pytorch_model.bin',
    'plain-5': 'data/dict1_epoch5.bin',
}
weights = loaded_models['plain-5']
model = models.get_model_for_api(weights_path=weights)


@app.route('/word', methods=["POST"])
def word():
    data = request.json
    word = data['word']

    definition = models.define(model, word, num_return=1)[0]
    definition.replace(']', '').replace('[', '')
    if definition[-1] != '.':
        definition += '.'

    resp_data = json.dumps(
            {'definition': definition}
        )
    response = Response(resp_data)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = True
    return response  # return data with 200 OK

@app.route('/', methods=["GET"])
def index():
    return redirect(f'http://hackathon.chrisdaw.net:3030', 301)

if __name__ == '__main__':
    """
    # Load ML model
    for k, weights_path in loaded_models.items():
        loaded_models[k] = models.get_model(weights_path)
    """

    app.debug=True
    app.run(host='0.0.0.0')  # run our Flask app