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
from datetime import datetime
from configs import production_config as pc, build_features_config as bfc
from threshold_finder import ThresholdFinder


def load_model():
    threshold_finder = ThresholdFinder()
    threshold_finder.import_threshold_from_file()
    threshold_comp = threshold_finder.to_list_comportamenti(flag_f1=False)
    threshold_day = threshold_finder.to_list_day(flag_f1=False)
    try:
        model = pickle.load(open(pc.machine_learning_model_path, 'rb'))
    except FileNotFoundError:
        print(f"File {pc.machine_learning_model_path} not found.  Aborting")
        sys.exit(1)
    return model, threshold_comp, threshold_day


def predict_model(input_model, model, threshold_comp, threshold_day):
    result, prediction = [], []
    try:
        prediction = model.predict_proba(input_model).tolist()
    except Exception:
        exit(1)
    index = input_model.index.values.tolist()
    for i, single_prediction in enumerate(prediction):
        current_e_index = single_prediction.index(max(single_prediction))
        if current_e_index < 2:
            pos_label = bfc.positive_target_comp
            thresholds = threshold_comp
        else:
            pos_label = bfc.positive_target_day
            thresholds = threshold_day
        prediction_value = float("{:.3f}".format(single_prediction[pos_label]))
        for j in range(len(thresholds)-1):
            if thresholds[j] < single_prediction[pos_label] <= thresholds[j+1]:
                result.append((index[i], prediction_value, j))
                break
    return result


def test(x=None, num_elements=100):
    start = datetime.now()
    if x is None:
        x = pd.read_csv(bfc.path_x_test, low_memory=False, index_col=[pc.index_name]).head(num_elements)
    model, threshold_comp, threshold_day = load_model()
    prediction = predict_model(x, model, threshold_comp, threshold_day)
    print(">> prediction: ", prediction)
    print("\n>> total time ", datetime.now() - start, "\n")


if __name__ == "__main__":
    test()
