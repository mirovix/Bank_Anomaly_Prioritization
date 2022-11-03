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
from flask_wtf.csrf import CSRFProtect
from flask import request, Response, jsonify
from configs import production_db_config as pdbc
from predict_anomaly import predict_model, load_model

app = flask.Flask(__name__)
csrf = CSRFProtect()
csrf.init_app(app)
model, threshold_comp, threshold_day = load_model()


def error(text, code):
    print("\n>> " + text + "\n")
    return Response(text, code)


@app.route(pdbc .web_service_app_route, methods=['GET'])
def prediction():
    try:
        x = pickle.loads(base64.b64decode(request.data))
    except FileNotFoundError:
        return error(f"file {request.data} not found.  Aborting", 400)
    except OSError:
        return error(f"os error occurred trying to open {request.files['data']}", 400)
    except Exception as err:
        return error((f"unexpected error opening {request.data} is", repr(err)), 400)
    return jsonify(predict_model(x, model, threshold_comp, threshold_day))


if __name__ == "__main__":
    app.config["DEBUG"] = False
    app.run()
