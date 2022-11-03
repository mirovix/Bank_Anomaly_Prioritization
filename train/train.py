#!/usr/bin/env python3

"""
@Author: Miro
@Date: 09/06/2022
@Version: 1.3
@Objective: training loop per allenare il modello
@TODO: soglia automatica del medio livello
"""

import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from configs import production_config as pc, train_config as tc
from load_dataset_split import load_dataset_split as load_ds
from models_definition import compile_model_voting, random_search_training
from performance_analysis import test_performance
from plots_train import plots_performance
from threshold_finder import ThresholdFinder, my_custom_loss_func


def smote_over_sampling(data, target, smote_model=tc.smote_model):
    print("\n>> smote implemented\n")
    x_resample, y_resample = smote_model.fit_resample(np.array(data.values.tolist()), np.array(target.values.tolist()))
    target = pd.DataFrame(y_resample, columns=target.columns)
    data = pd.DataFrame(x_resample, columns=data.columns)
    return data, target


def model_definition_search(x_train, y_train):
    print(">> start random search train\n")
    score = make_scorer(my_custom_loss_func, greater_is_better=True, needs_proba=True)
    rf_random = random_search_training(score=score)
    rf_random.fit(x_train, y_train.values.ravel())
    print(">> best parameters ", rf_random.best_params_)
    with open(tc.best_parameters_directory, 'w') as f:
        f.write(str(rf_random.best_params_))
    return rf_random


def model_definition(x_train, y_train):
    model = compile_model_voting()
    model.fit(x_train, y_train.values.ravel())
    return model


def train():
    start = datetime.now()

    x_train_orig, y_train_orig, x_val, y_val, x_test, y_test = load_ds()
    y_val_arr = np.array(y_val).reshape(len(y_val), )
    y_test_arr = np.array(y_test).reshape(len(y_test), )

    if tc.apply_smote is True:
        x_train, y_train = smote_over_sampling(x_train_orig, y_train_orig)
    else:
        x_train, y_train = x_train_orig, y_train_orig

    if tc.apply_random_search is True:
        model = model_definition_search(x_train, y_train)
    else:
        model = model_definition(x_train, y_train)

    y_pred_perc = model.predict_proba(x_val)
    y_pred_perc_test = model.predict_proba(x_test)

    thresholds = ThresholdFinder(y_pred_perc, y_val_arr, mid_threshold_fix=True)
    thresholds.define_thresholds()

    plots_performance(y_pred_perc_test, y_test_arr, thresholds)

    test_performance(y_pred_perc_test, y_test_arr, thresholds)

    pickle.dump(model, open(pc.machine_learning_model_path, 'wb'))

    print("\n>> total time for train ", datetime.now() - start, "\n")


if __name__ == "__main__":
    train()
