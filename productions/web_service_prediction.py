#!/usr/bin/env python3

"""
@Author: Miro
@Date: 12/09/2022
@Version: 1.1
@Objective: web service in gestione delle chiamate per la predizione del modello
@TODO:
"""

import base64
import pickle
import flask
from flask import request, Response
from predict_anomaly import predict_model, load_model

app = flask.Flask(__name__)
model, thresholds = load_model()


def error(text, code):
    print("\n>> " + text + "\n")
    return Response(text, code)


@app.route('/prediction', methods=['GET'])
def prediction():
    try:
        x = pickle.loads(base64.b64decode(request.data))
    except FileNotFoundError:
        return error(f"file {request.data} not found.  Aborting", 400)
    except OSError:
        return error(f"os error occurred trying to open {request.files['data']}", 400)
    except Exception as err:
        return error((f"unexpected error opening {request.data} is", repr(err)), 400)
    return predict_model(x, model, thresholds)


if __name__ == "__main__":
    app.config["DEBUG"] = True
    app.run()