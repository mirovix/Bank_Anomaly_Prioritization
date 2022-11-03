#!/usr/bin/env python3

"""
@Author: Miro
@Date: 14/10/2022
@Version: 1.1
@Objective: definire le differenti soglie
@TODO:
"""

import json
import numpy as np
from sklearn.metrics import f1_score
from configs import production_config as pc, build_features_config as bfc, train_config as tc
from util import estimation_single_anomaly as esa


def my_custom_loss_func(y_true, y_pred, step=0.01):
    f1_comportamenti_threshold, f1_day_threshold = 0, 0
    for threshold in np.arange(0.05, 1, step):
        y_pred_ = from_threshold_to_pred(y_pred, threshold, threshold)
        f1 = f1_score(y_true, y_pred_, average=None).tolist()
        if f1[bfc.positive_target_comp] >= f1_comportamenti_threshold: f1_comportamenti_threshold = f1[bfc.positive_target_comp]
        if f1[bfc.positive_target_day] >= f1_day_threshold: f1_day_threshold = f1[bfc.positive_target_day]
    return (f1_comportamenti_threshold + f1_day_threshold) / 2


def from_threshold_to_pred(y_pred, threshold_comp, threshold_day):
    y_pred_list = []
    for e in y_pred.tolist():
        if e.index(max(e)) < bfc.negative_target_day:
            if e[bfc.positive_target_comp] >= threshold_comp:
                y_pred_list.append(bfc.positive_target_comp)
            else:
                y_pred_list.append(bfc.negative_target_comp)
        else:
            if e[bfc.positive_target_day] >= threshold_day:
                y_pred_list.append(bfc.positive_target_day)
            else:
                y_pred_list.append(bfc.negative_target_day)
    return np.array(y_pred_list)


class ThresholdFinder:
    def __init__(self, x_val=None, y_val=None, step_f1=0.01, step_rmv=0.00001, safe_removing=False,
                 mid_threshold_fix=False, path=pc.machine_learning_thresholds_data_path):
        self.x_val = x_val
        self.y_val = y_val

        self.step_f1 = step_f1
        self.step_rmv = step_rmv

        self.safe_removing = safe_removing
        self.mid_threshold_fix = mid_threshold_fix

        self.path_to_save = path
        self.name_json_comportamenti = 'threshold_comp.json'
        self.name_json_day = 'threshold_day.json'

        self.rmv_comp, self.rmv_day = None, None
        self.lower_comp, self.lower_day = None, None
        self.upper_comp, self.upper_day = None, None
        self.f1_comp, self.f1_day = None, None

    def define_thresholds(self):
        self.validation_f1_threshold()
        self.validation_for_removing_threshold()
        self.thresholds_range()
        self.print_threshold()
        self.write_list_threshold()

    def validation_f1_threshold(self):
        f1_day, f1_comp, self.f1_day, self.f1_comp = 0, 0, 0, 0
        for threshold in np.arange(0, 1, self.step_f1):
            y_pred = from_threshold_to_pred(self.x_val, threshold, threshold)
            f1 = f1_score(self.y_val, y_pred, average=None).tolist()
            if f1[bfc.positive_target_comp] >= f1_comp:
                f1_comp = f1[bfc.positive_target_comp]
                self.f1_comp = threshold
            if f1[bfc.positive_target_day] >= f1_day:
                f1_day = f1[bfc.positive_target_day]
                self.f1_day = threshold

    def validation_for_removing_threshold(self):
        self.rmv_day, self.rmv_comp = 0.0, 0.0
        if self.safe_removing is True: return
        for threshold in np.arange(0, 1, self.step_rmv):
            to_remove_info, _, _ = esa(self.y_val, self.x_val, threshold, threshold, validation=True)
            if to_remove_info['comp'] == 0: self.rmv_comp = threshold
            if to_remove_info['day'] == 0: self.rmv_day = threshold
            if to_remove_info['err_day'] + to_remove_info['err_comp'] > 1: break

    def thresholds_range(self):
        if self.mid_threshold_fix is True:
            self.lower_day, self.upper_day = tc.fixed_threshold_mid
            self.lower_comp, self.upper_comp = tc.fixed_threshold_mid
            return

        _, wrong_pred, _ = esa(self.y_val, self.x_val, self.rmv_comp,
                               self.rmv_day, self.f1_comp, self.f1_comp)

        wrong_pred_comp = wrong_pred['comp_fn'] + wrong_pred['comp_fp']
        wrong_pred_comp.sort()

        wrong_pred_day = wrong_pred['day_fn'] + wrong_pred['day_fp']
        wrong_pred_day.sort()

        self.lower_comp = wrong_pred_comp[0]
        self.upper_comp = wrong_pred_comp[len(wrong_pred_comp) - 1]

        self.lower_day = wrong_pred_day[0]
        self.upper_day = wrong_pred_day[len(wrong_pred_day) - 1]

    def print_threshold(self):
        print("\n>> comportamenti")
        print("  >> threshold remove " + str(self.rmv_comp))
        print("  >> threshold lower " + str(self.lower_comp))
        print("  >> threshold f1 " + str(self.f1_comp))
        print("  >> threshold upper " + str(self.upper_comp))
        print("\n>> day")
        print("  >> threshold remove " + str(self.rmv_day))
        print("  >> threshold lower " + str(self.lower_day))
        print("  >> threshold f1 " + str(self.f1_day))
        print("  >> threshold upper " + str(self.upper_day))

    def to_list_comportamenti(self, flag_f1=True):
        if flag_f1 is True:
            return [-0.01, self.rmv_comp, self.lower_comp, self.f1_comp, self.upper_comp, 1]
        else:
            return [-0.01, self.rmv_comp, self.lower_comp, self.upper_comp, 1]

    def to_list_day(self, flag_f1=True):
        if flag_f1 is True:
            return [-0.01, self.rmv_day, self.lower_day, self.f1_day, self.upper_day, 1]
        else:
            return [-0.01, self.rmv_day, self.lower_day, self.upper_day, 1]

    def import_threshold_from_file(self):
        comp_list, day_list = self.read_list_thresholds()
        _, self.rmv_day, self.lower_day, self.f1_day, self.upper_day, _ = day_list
        _, self.rmv_comp, self.lower_comp, self.f1_comp, self.upper_comp, _ = comp_list

    def read_list_thresholds(self):
        with open(self.path_to_save + self.name_json_day, 'rb') as fp:
            day_list = json.load(fp)
        with open(self.path_to_save + self.name_json_comportamenti, 'rb') as fp:
            comp_list = json.load(fp)
        return comp_list, day_list

    def write_list_threshold(self):
        with open(self.path_to_save + self.name_json_comportamenti, 'w') as fp:
            json.dump(self.to_list_comportamenti(), fp)
        with open(self.path_to_save + self.name_json_day, 'w') as fp:
            json.dump(self.to_list_day(), fp)
