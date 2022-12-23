#!/usr/bin/env python3

"""
@Author: Miro
@Date: 15/06/2022
@Version: 1.3
@Objective: caricamento del dataset diviso in train, validation e test
@TODO:
"""

import pandas as pd
from categorization import Categorization
from configs import build_features_config as bfc, production_config as pc


def split_dataset():
    data, target = Categorization().run()
    x_train, x_val, x_test = data
    y_train, y_val, y_test = target

    x_train.to_csv(bfc.path_x_train)
    y_train.to_csv(bfc.path_y_train, index=False)

    x_val.to_csv(bfc.path_x_val)
    y_val.to_csv(bfc.path_y_val, index=False)

    x_test.to_csv(bfc.path_x_test)
    y_test.to_csv(bfc.path_y_test, index=False)

    print("\n>> data are split\n")


def load_dataset_split(test=False):
    x_test = pd.read_csv(bfc.path_x_test, low_memory=False, index_col=[pc.index_name])
    y_test = pd.read_csv(bfc.path_y_test, low_memory=False)
    print("\n>> test data are loaded from " + bfc.path_x_test + " - " + bfc.path_y_test)
    if test is True:
        return x_test, y_test

    x_train_orig = pd.read_csv(bfc.path_x_train, low_memory=False, index_col=[pc.index_name])
    y_train_orig = pd.read_csv(bfc.path_y_train, low_memory=False)
    print("\n>> train data are loaded from " + bfc.path_x_train + " - " + bfc.path_y_train)

    x_val = pd.read_csv(bfc.path_x_val, low_memory=False, index_col=[pc.index_name])
    y_val = pd.read_csv(bfc.path_y_val, low_memory=False)
    print("\n>> validation data are loaded from " + bfc.path_x_val + " - " + bfc.path_y_val)

    print("\n>> size train ", x_train_orig.shape[0])
    print(">> size validation ", x_val.shape[0])
    print(">> size test ", x_test.shape[0])
    print("\n")

    return x_train_orig, y_train_orig, x_val, y_val, x_test, y_test


if __name__ == "__main__":
    split_dataset()
    exit(0)
