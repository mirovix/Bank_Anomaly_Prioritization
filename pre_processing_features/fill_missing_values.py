#!/usr/bin/env python3

"""
@Author: Miro
@Date: 09/06/2022
@Version: 1.11
@Objective: completare i valori mancanti all'interno del dataset
@TODO:
"""

import pandas as pd
from configs import build_features_config as bfc


class FillMissingValues:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def run(self):

        self.round_values()
        self.x.insert(0, bfc.y_col_name, self.y[bfc.y_col_name].values.tolist(), True)
        self.x = self.x.dropna()

        self.y = pd.DataFrame(data=self.x[bfc.y_col_name], columns=[bfc.y_col_name])
        self.x = self.x.drop(columns=bfc.y_col_name)

        return self.x, self.y

    def run_production(self):
        self.round_values()
        self.x = self.x.dropna()
        return self.x

    def round_values(self):
        for e in bfc.dtypes_num:
            self.x[e] = self.x[e].astype(pd.Int64Dtype())
        for e in bfc.dtypes_cat:
            self.x[e] = self.x[e].astype(str)
