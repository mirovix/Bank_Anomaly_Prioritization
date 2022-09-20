#!/usr/bin/env python3

"""
@Author: Miro
@Date: 09/06/2022
@Version: 1.1
@Objective: completare i valori mancanti all'interno del dataset
@TODO:
"""

import pandas as pd
import missingno as msn
import matplotlib.pyplot as plt
import numpy as np


class FillMissingValues:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def run(self):

        y_col_name = "EVALUATION"
        self.round_values()
        self.x.insert(0, y_col_name, self.y[y_col_name].values.tolist(), True)
        self.x = self.x.dropna()

        self.y = pd.DataFrame(data=self.x[y_col_name],  columns=[y_col_name])
        self.x = self.x.drop(columns=y_col_name)

        return self.x, self.y

    def run_production(self):
        self.round_values()
        self.x = self.x.dropna()
        return self.x

    def round_values(self):
        dtypes_num = ['NCHECKREQUIRED', 'NCHECKDEBITED', 'NCHECKAVAILABLE', 'RISK_PROFILE']
        dtypes_cat = ['REPORTED']
        for e in dtypes_num:
            self.x[e] = self.x[e].astype(pd.Int64Dtype())
        for e in dtypes_cat:
            self.x[e] = self.x[e].astype(str)
