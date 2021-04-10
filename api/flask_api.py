from flask import Flask, Response, jsonify, request, redirect
from flask_cors import CORS
import pandas as pd
import json
import sys

app = Flask(__name__)
cors = CORS(app)


@app.route('/word', methods=["POST"])
def word():
    data = request.json
    print(f"user entered {data['word']}")
    resp_data = json.dumps({'definition': f"{data['word']} but from python"})
    response = Response(resp_data)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = True
    return response  # return data with 200 OK

@app.route('/', methods=["GET"])
def index():
    return redirect(f'http://hackathon.chrisdaw.net:3030', 301)

if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0')  # run our Flask app