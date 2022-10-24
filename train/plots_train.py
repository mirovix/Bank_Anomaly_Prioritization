#!/usr/bin/env python3

"""
@Author: Miro
@Date: 14/10/2022
@Version: 1.1
@Objective: gestione dei plot del train
@TODO:
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from configs import build_features_config as bfc
from functions_plot import plot_table_soglie, plot_confusion_matrix, plot_roc
from threshold_finder import from_threshold_to_pred


def compute_soglie_errors(y_model_i, y_target_i, th_list, pos_target, neg_target, soglie_errors):
    for j in range(len(th_list) - 1):
        target = neg_target if j > 2 else pos_target
        k = 1 if j > 2 else 0
        t = j - 1 if j > 2 else j
        if th_list[j] < y_model_i[pos_target] <= th_list[j + 1] and (y_target_i == target):
            soglie_errors[k][t] += 1
            break


def compute_soglie_predictions(y_model_i, y_target_i, th_list_without_f1, pos_target, neg_target, soglie_predictions):
    condition = (y_target_i == neg_target or y_target_i == pos_target)
    for j in range(len(th_list_without_f1) - 1):
        if th_list_without_f1[j] < y_model_i[pos_target] <= th_list_without_f1[j + 1] and condition:
            soglie_predictions[j] += 1
            break


def distribution_anomaly(y_model, y_target, thresholds, day=False):
    soglie_errors = [[0, 0, 0, None], [None, None, 0, 0]]
    soglie_predictions = [0, 0, 0, 0]

    if day is False:
        pos_target, neg_target, name = bfc.positive_target_comp, bfc.negative_target_comp, bfc.name_comp
        th_list_without_f1 = thresholds.to_list_comportamenti(flag_f1=False)
        th_list = thresholds.to_list_comportamenti()
    else:
        pos_target, neg_target, name = bfc.positive_target_day, bfc.negative_target_day, bfc.name_day
        th_list_without_f1 = thresholds.to_list_day(flag_f1=False)
        th_list = thresholds.to_list_day()

    for i in range(y_target.shape[0]):
        compute_soglie_predictions(y_model[i], y_target[i], th_list_without_f1, pos_target, neg_target, soglie_predictions)
        compute_soglie_errors(y_model[i], y_target[i], th_list, pos_target, neg_target, soglie_errors)

    return soglie_predictions, soglie_errors, th_list_without_f1, name


def roc_process_plot(y_pred, y_test_arr, day=False):
    if day is True:
        condition, positive_target = y_pred < 2, bfc.positive_target_day
    else:
        condition, positive_target = y_pred >= 2, bfc.positive_target_comp
    y_pred_temp = np.delete(y_pred, np.where(condition))
    y_test_arr_temp = np.delete(y_test_arr, np.where(condition))
    plot_roc(y_pred_temp, y_test_arr_temp, positive_target)


def plots_performance(y_pred_perc, y_test_arr, thresholds):
    y_pred = from_threshold_to_pred(y_pred_perc, thresholds.f1_comp, thresholds.f1_day)

    print("\n>> model accuracy ", accuracy_score(y_test_arr, y_pred))

    plot_confusion_matrix(confusion_matrix(y_test_arr, y_pred), classes=[0, 1, 2, 3])

    print("\n>> general evaluation ", str(precision_recall_fscore_support(y_test_arr, y_pred)))

    roc_process_plot(y_pred, y_test_arr)
    roc_process_plot(y_pred, y_test_arr, day=True)

    # plot_feature_importance(classifier.feature_importances_, x_test.columns)

    plot_table_soglie(distribution_anomaly(y_pred_perc, y_test_arr, thresholds))
    plot_table_soglie(distribution_anomaly(y_pred_perc, y_test_arr, thresholds, day=True))
