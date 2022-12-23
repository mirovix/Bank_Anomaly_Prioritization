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
from sys import exit
from datetime import datetime
from configs import production_config as pc, build_features_config as bfc
from train.threshold_finder import ThresholdFinder


def load_model():
    threshold_finder = ThresholdFinder()
    threshold_finder.import_threshold_from_file()
    threshold_comp = threshold_finder.to_list_comportamenti(flag_f1=False)
    threshold_day = threshold_finder.to_list_day(flag_f1=False)
    try:
        input_model = open(pc.machine_learning_model_path, 'rb')
        model = pickle.load(input_model)
        input_model.close()
    except FileNotFoundError:
        print(f">> file {pc.machine_learning_model_path} not found")
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
        prediction_value = float("{:.4f}".format(single_prediction[pos_label]))
        for j in range(len(thresholds)-1):
            if thresholds[j] < single_prediction[pos_label] <= thresholds[j+1]:
                result.append((index[i], prediction_value, j))
                break
    return result


def test_prediction(x=None, num_elements=100, print_prediction=False):
    start = datetime.now()
    if x is None:
        x = pd.read_csv(bfc.path_x_test, low_memory=False, index_col=[pc.index_name]).head(num_elements)
    model, threshold_comp, threshold_day = load_model()
    prediction = predict_model(x, model, threshold_comp, threshold_day)
    prediction_df = pd.DataFrame(prediction, columns=pc.col_name) \
                    .astype(pc.dtype_col) \
                    .sort_values(pc.col_name, ascending=True) \
                    .drop_duplicates(subset=[pc.index_name], keep="last")
    if print_prediction is True:
        print(">> prediction: ", prediction)
        print("\n>> total time ", datetime.now() - start, "\n")
    return prediction_df


if __name__ == "__main__":
    test_prediction(print_prediction=True)
