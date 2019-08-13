# importing libraries
import os
import numpy as np
import flask
from flask import Flask, render_template, request
from flask_cors import CORS
from predictor import makePrediction

# creating instance of the class
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/prediction', methods=['POST'])
def result():
    if request.method == 'POST':
        image_data = request.data
        result = makePrediction(image_data)
        return result, 200
