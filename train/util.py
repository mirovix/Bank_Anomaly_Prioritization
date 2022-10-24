#!/usr/bin/env python3

"""
@Author: Miro
@Date: 14/10/2022
@Version: 1.1
@Objective: funzioni di supporto per il train
@TODO:
"""

from configs import build_features_config as bfc


def choose_threshold_target(comp_th, day_th, j):
    return comp_th if j < bfc.negative_target_day else day_th, bfc.positive_target_comp \
        if j < bfc.negative_target_day else bfc.positive_target_day


def condition_target_threshold(y_i, threshold, target, j, wrong_sign=False, correct_sign=False):
    if wrong_sign is True:
        if target != j:
            return y_i[target] > threshold
        else:
            return y_i[target] < threshold
    elif correct_sign is True:
        if target != j:
            return y_i[target] < threshold
        else:
            return y_i[target] > threshold
    else:
        return y_i[target] <= threshold


def estimate(y_model_i, y_target_i, dictionary, threshold_comp, threshold_day, print_info=False,
             wrong_sign=False, correct_sign=False):
    for j, key in enumerate(dictionary.keys()):
        threshold, target = choose_threshold_target(threshold_comp, threshold_day, j)
        if condition_target_threshold(y_model_i, threshold, target, j, wrong_sign, correct_sign) and y_target_i == j:
            dictionary[key].append(y_model_i[target])
            if print_info is True: print(" >> " + key + " >> y_model " + str(y_model_i) + " y_target " + str(y_target_i))
            break
    return dictionary


def estimation_single_anomaly(y_target, y_model, threshold_rmv_comp=0.0, threshold_rmv_day=0.0, threshold_comp=0.0,
                              threshold_day=0.0, validation=False, print_info=False):

    to_remove_dict = {'comp': [], 'err_day': [], 'day': [], 'err_comp': []}
    wrong_dict = {'comp_fp': [], 'comp_fn': [], 'day_fp': [], 'day_fn': []}
    correct_dict = {'comp_tp': [], 'comp_tn': [], 'day_tp': [], 'day_tn': []}

    for i in range(y_target.shape[0]):

        to_remove_dict.update(estimate(y_model[i], y_target[i], to_remove_dict, threshold_rmv_comp, threshold_rmv_day))

        if validation is True: continue

        wrong_dict.update(estimate(y_model[i], y_target[i], wrong_dict, threshold_comp, threshold_day, print_info=print_info, wrong_sign=True))

        correct_dict.update(estimate(y_model[i], y_target[i], correct_dict, threshold_comp, threshold_day, correct_sign=True))

    for key in to_remove_dict: to_remove_dict[key] = len(to_remove_dict[key])
    return to_remove_dict, wrong_dict, correct_dict
