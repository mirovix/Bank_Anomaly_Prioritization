#!/usr/bin/env python3

"""
@Author: Miro
@Date: 07/09/2022
@Version: 1.0
@Objective: analisi degli errori sul test
@TODO:
"""

import pickle
import numpy as np
import pandas as pd
from load_dataset_split import load_dataset_split as load_data
from threshold_finder import ThresholdFinder
from configs import build_features_config as bfc, train_config as tc, production_config as pc


def check_range(y_pred, y_test, threshold, f1_th, name, index, n_target, p_target, errors):
    if y_pred > threshold[1] and y_test == n_target:
        errors.append([index, y_pred * 100, tc.da_segnalare_name, tc.soglie[3], tc.da_non_segnalare_name, name])
    elif f1_th < y_pred <= threshold[1] and y_test == n_target:
        errors.append([index, y_pred * 100, tc.da_segnalare_name, tc.soglie[2], tc.da_non_segnalare_name, name])
    elif threshold[0] < y_pred <= f1_th and y_test == p_target:
        errors.append([index, y_pred * 100, tc.da_non_segnalare_name, tc.soglie[2], tc.da_segnalare_name, name])
    elif y_pred <= threshold[0] and y_test == p_target:
        errors.append([index, y_pred * 100, tc.da_non_segnalare_name, tc.soglie[1], tc.da_segnalare_name, name])


def import_threshold():
    th_finder = ThresholdFinder()
    th_finder.import_threshold_from_file()
    target_list = [[bfc.negative_target_comp, bfc.positive_target_comp, th_finder.f1_comp,
                    [th_finder.lower_comp, th_finder.upper_comp]],
                   [bfc.negative_target_day, bfc.positive_target_day, th_finder.f1_day,
                    [th_finder.lower_day, th_finder.upper_day]]]
    return target_list


def error_estimation():
    target_list = import_threshold()

    x_test, y_test = load_data(test=True)
    loaded_model = pickle.load(open(pc.machine_learning_model_path, 'rb'))

    y_pred = loaded_model.predict_proba(x_test)
    y_test = np.array(y_test).reshape(len(y_test), )

    index_list, errors = x_test.index.values.tolist(), []
    for i in range(y_test.shape[0]):
        for n_target, p_target, f1_th, threshold in target_list:
            name = bfc.name_comp if p_target == bfc.positive_target_comp else bfc.name_day
            check_range(y_pred[i][p_target], y_test[i], threshold, f1_th, name, index_list[i], n_target, p_target, errors)

    pd.DataFrame(errors, columns=tc.col_names).to_csv(tc.path_save_errors, index=False)
    print(">> performance of the errors are saved in " + tc.path_save_errors)


if __name__ == "__main__":
    error_estimation()
