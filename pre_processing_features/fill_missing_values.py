#!/usr/bin/env python3

"""
@Author: Miro
@Date: 09/06/2022
@Version: 1.0
@Objective: completare i valori mancanti all'interno del dataset
@TODO:
"""

import pandas as pd
import missingno as msn
import matplotlib.pyplot as plt
import numpy as np
from load_dataset import load_dataset


class FillMissingValues:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def run(self, plot=True, info=True, corr_mat=True):

        if corr_mat is True:
            self.correlation_matrix()
        if info is True:
            self.show_info()
        if plot is True:
            self.plot_info()

        y_col_name = "EVALUATION"
        self.round_values()
        self.x.insert(0, y_col_name, self.y[y_col_name].values.tolist(), True)
        self.x = self.x.dropna()

        self.x = self.x.reset_index(drop=True)

        self.y = pd.DataFrame(data=self.x[y_col_name],  columns=[y_col_name])
        self.x = self.x.drop(columns=y_col_name)

        return self.x, self.y

    def round_values(self):
        dtypes_num = ['NCHECKREQUIRED', 'NCHECKDEBITED', 'NCHECKAVAILABLE', 'RISK_PROFILE']
        dtypes_cat = ['REPORTED']
        for e in dtypes_num:
            self.x[e] = self.x[e].astype(pd.Int64Dtype())
        for e in dtypes_cat:
            self.x[e] = self.x[e].astype(str)

    def correlation_matrix(self, path_to_save="data/corr_matrix/corr_matrix.csv"):
        df = self.x.iloc[:, [i for i, n in enumerate(np.var(self.x.isnull(), axis='rows')) if n > 0]]
        corr_mat = df.isnull().corr()
        corr_mat.to_csv(path_to_save, index=False)
        print(">> correlation matrix is saved in ", path_to_save)
        return corr_mat

    def show_info(self):
        print(">> GENERAL INFORMATION")
        self.x.info()
        print(">> NULL VALUES COUNT")
        print(self.x.isna().sum())

    def plot_info(self):
        msn.matrix(self.x)
        plt.show()
        msn.bar(self.x)
        plt.show()
        msn.heatmap(self.x)
        plt.show()


# data_x, data_y, _ = load_dataset()
# missing_values = FillMissingValues(data_x, data_y).run(False, False, False)
