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
    '1': 'data/dict-short.bin',
    '2': 'data/dict-long.bin',
    '3': 'data/urbandict.bin',
    '4': 'data/urbandict-long.bin',
}
model = input(f"{loaded_models}\nEnter model number from above: ")
weights = loaded_models[model]
model = models.get_model_for_api(weights_path=weights)


@app.route('/word', methods=["POST"])
def word():
    data = request.json
    word = data['word']

    try:
        definition = models.define(model, word, num_return=1)[0]
        definition = definition.replace(']', '').replace('[', '')\
                        .replace('fuck', 'duck').replace('cunt', 'trunk')\
                        .replace('sex', 'love').replace('genital', 'appendage')
        if definition[-1] != '.':
            definition += '.'
    except:
        definition = "This word is undefinable. Good job..."

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