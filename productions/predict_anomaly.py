#!/usr/bin/env python3

"""
@Author: Miro
@Date: 12/09/2022
@Version: 1.1
@Objective: caricamento del modello e predizione delle anomalie
@TODO:
"""

import pickle
import sys
import pandas as pd
from flask import jsonify


def check_input_features(to_predict):
    return to_predict


def load_model(filename_model='../train/models/voting_model.sav', thresholds=None):
    if thresholds is None:
        thresholds = [0, 0.20, 0.60]
    try:
        model = pickle.load(open(filename_model, 'rb'))
    except FileNotFoundError:
        print(f"File {filename_model} not found.  Aborting")
        sys.exit(1)
    except Exception as err:
        print(f"Unexpected error predicting anomalies is" + repr(err))
        sys.exit(1)
    return model, thresholds


def predict_model(to_predict, model, thresholds, test=False):
    input_model, result = check_input_features(to_predict), []
    for i in range(input_model.shape[0]):
        index = input_model.index.values.tolist()[i]
        try:
            single_prediction = model.predict_proba(input_model.iloc[i]).tolist()
        except:
            result.append((index, -1, -1))
            continue
        current_e_index = single_prediction.index(max(single_prediction))
        pos_label = 1 if current_e_index < 2 else 3
        prediction = float("{:.3f}".format(single_prediction[pos_label]))
        if thresholds[1] < single_prediction[pos_label] <= thresholds[2]:
            result.append((index, prediction, 2))
        elif single_prediction[pos_label] <= thresholds[0]:
            result.append((index, prediction, 0))
        elif single_prediction[pos_label] > thresholds[2]:
            result.append((index, prediction, 3))
        elif thresholds[0] < single_prediction[pos_label] <= thresholds[1]:
            result.append((index, prediction, 1))

    if test is True:
        return result
    return jsonify(result)


def test(x=None, path="../data/dataset_test/test.csv"):
    if x is None:
        x = pd.read_csv(path, low_memory=False, index_col=['ID'])
    model, thresholds = load_model()
    prediction = predict_model(x, model, thresholds, True)
    print("Prediction: ", prediction)


if __name__ == "__main__":
    test()
