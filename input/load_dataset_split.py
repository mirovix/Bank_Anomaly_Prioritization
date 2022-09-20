#!/usr/bin/env python3

"""
@Author: Miro
@Date: 15/06/2022
@Version: 1.22
@Objective: caricamento del dataset diviso in train, validation e test
@TODO:
"""

import pandas as pd
from categorization import Categorization


def split_dataset(path_x_train="../data/dataset_split/x_train.csv", path_y_train="../data/dataset_split/y_train.csv",
                  path_x_val="../data/dataset_split/x_val.csv", path_y_val="../data/dataset_split/y_val.csv",
                  path_x_test="../data/dataset_split/x_test.csv", path_y_test="../data/dataset_split/y_test.csv"):
    data, target = Categorization().run()
    x_train, x_val, x_test = data
    y_train, y_val, y_test = target

    x_train.to_csv(path_x_train)
    y_train.to_csv(path_y_train, index=False)

    x_val.to_csv(path_x_val)
    y_val.to_csv(path_y_val, index=False)

    x_test.to_csv(path_x_test)
    y_test.to_csv(path_y_test, index=False)

    print("\n>> data are split")


def load_dataset_split(path_x_train="data/dataset_split/x_train.csv", path_y_train="data/dataset_split/y_train.csv",
                       path_x_val="data/dataset_split/x_val.csv", path_y_val="data/dataset_split/y_val.csv",
                       path_x_test="data/dataset_split/x_test.csv", path_y_test="data/dataset_split/y_test.csv",
                       test=False):
    x_test = pd.read_csv(path_x_test, low_memory=False, index_col=['ID'])
    y_test = pd.read_csv(path_y_test, low_memory=False)
    print("\n>> test data are loaded from " + path_x_test + " and " + path_y_test)
    if test is True:
        return x_test, y_test

    x_train_orig = pd.read_csv(path_x_train, low_memory=False, index_col=['ID'])
    y_train_orig = pd.read_csv(path_y_train, low_memory=False)
    print("\n>> train data are loaded from " + path_x_train + " and " + path_y_train)

    x_val = pd.read_csv(path_x_val, low_memory=False, index_col=['ID'])
    y_val = pd.read_csv(path_y_val, low_memory=False)
    print("\n>> validation data are loaded from " + path_x_val + " and " + path_y_val)

    print("\n>> size train ", x_train_orig.shape[0])
    print(">> size validation ", x_val.shape[0])
    print(">> size test ", x_test.shape[0])

    return x_train_orig, y_train_orig, x_val, y_val, x_test, y_test


if __name__ == "__main__":
    split_dataset()
