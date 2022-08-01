#!/usr/bin/env python3

"""
@Author: Miro
@Date: 15/06/2022
@Version: 1.2
@Objective: caricamento del dataset
@TODO:
"""

import pandas as pd


def load_dataset(path_x="data/dataset/dataset_x.csv", path_y="data/dataset/dataset_y.csv",
                 path_x_evaluated="data/dataset/dataset_x_evaluated.csv"):

    x_dataset = pd.read_csv(path_x, low_memory=False)
    print("\n>> dataset_x is loaded from ", path_x)
    if path_y is None and path_x_evaluated is None:
        return x_dataset

    y_dataset = pd.read_csv(path_y)
    print(">> dataset_y is loaded from ", path_y)
    if path_x_evaluated is None:
        return x_dataset, y_dataset

    x_eval_dataset = pd.read_csv(path_x_evaluated, low_memory=False)
    print(">> dataset x_evaluated is loaded from ", path_x_evaluated)

    return x_dataset, y_dataset, x_eval_dataset
