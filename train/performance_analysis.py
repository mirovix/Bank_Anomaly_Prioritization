#!/usr/bin/env python3

"""
@Author: Miro
@Date: 14/10/2022
@Version: 1.1
@Objective: gestione e analisi delle performance
@TODO:
"""

import numpy as np
from functions_plot import weights_definition, plot_wrong_predictions, plot_predictions
from configs import build_features_config as bfc
from util import estimation_single_anomaly as esa


def print_single_anomaly_comparison(y_target, to_remove_info):
    unique, counts = np.unique(y_target, return_counts=True)
    dict_unique = dict(zip(unique, counts))
    num_target_comp = dict_unique[bfc.negative_target_comp] + dict_unique[bfc.positive_target_comp]
    num_target_day = dict_unique[bfc.negative_target_day] + dict_unique[bfc.positive_target_day]

    print("\n>> removing analysis")
    for i, key in enumerate(to_remove_info):
        perc_value = str(to_remove_info[key] / (num_target_comp if i < 2 else num_target_day) * 100)
        print("     >> " + key + " with # elements " + str(to_remove_info[key]) + " >> " + perc_value + " %")

    sum_to_remove = to_remove_info['day'] + to_remove_info['comp']
    perc_value = str(sum_to_remove / y_target.shape[0] * 100)
    print("     >> comp+day with # of elements " + str(sum_to_remove) + " >> " + perc_value + " %")


def test_performance(y_model, y_target, thresholds):
    np.set_printoptions(suppress=True)
    to_remove_info, wrong_pred, correct_pred = esa(y_target, y_model,
                                                   thresholds.rmv_comp, thresholds.rmv_day,
                                                   thresholds.f1_comp, thresholds.f1_day)

    print_single_anomaly_comparison(y_target, to_remove_info)

    th_no_f1_day = thresholds.to_list_day(flag_f1=False)
    th_no_f1_comp = thresholds.to_list_comportamenti(flag_f1=False)

    weights_comp = weights_definition(correct_pred['comp_tp'], correct_pred['comp_tn'], wrong_pred['comp_fn'],
                                      wrong_pred['comp_fp'])
    weights_day = weights_definition(correct_pred['day_tp'], correct_pred['day_tn'], wrong_pred['day_fn'],
                                     wrong_pred['day_fp'])

    wrong_input_comp = (wrong_pred['comp_fn'], wrong_pred['comp_fp'],
                        th_no_f1_comp, thresholds.f1_comp, bfc.name_comp, weights_comp)
    wrong_input_day = (wrong_pred['day_fn'], wrong_pred['day_fp'],
                       th_no_f1_day, thresholds.f1_day, bfc.name_day, weights_day)
    correct_input_comp = (correct_pred['comp_tp'], correct_pred['comp_tn'],
                          th_no_f1_comp, thresholds.f1_comp, bfc.name_comp, weights_comp)
    correct_input_day = (correct_pred['day_tp'], correct_pred['day_tn'],
                         th_no_f1_day, thresholds.f1_day, bfc.name_day, weights_day)

    plot_wrong_predictions(wrong_input_comp)
    plot_wrong_predictions(wrong_input_day)
    plot_predictions(correct_input_comp)
    plot_predictions(correct_input_day)
