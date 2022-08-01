#!/usr/bin/env python3

"""
@Author: Miro
@Date: 09/06/2022
@Version: 1.0
@Objective: categorizzazione delle features
@TODO:
"""

import pandas as pd
from fill_missing_values import FillMissingValues
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from load_dataset import load_dataset


class Categorization:
    def __init__(self, show_info=False):
        self.data, self.target, self.data_eval = load_dataset()
        self.data, self.target = FillMissingValues(self.data, self.target).run(show_info, show_info, show_info)
        self.data_anomaly_list = self.data.DATA.values.tolist()
        self.data = self.data.drop(columns='DATA')
        self.data_eval = self.data_eval.dropna()

    def run(self, cat=False, cat_num=False):
        if cat is True:
            return self.preprocess_cat(self.data), self.target, self.data_anomaly_list, self.preprocess_cat(self.data_eval)
        if cat_num is True:
            return self.preprocess_num_cat(self.data), self.target, self.data_anomaly_list, self.preprocess_num_cat(self.data_eval)

    def num_preprocess(self):
        numerical_columns_selector = selector(dtype_exclude=object)
        numerical_columns = numerical_columns_selector(self.data)
        numerical_preprocessor = MinMaxScaler()
        return numerical_columns, numerical_preprocessor

    def cat_preprocess(self):
        categorical_columns_selector = selector(dtype_include=object)
        categorical_columns = categorical_columns_selector(self.data)
        categorical_preprocessor = OneHotEncoder(sparse=False)
        return categorical_columns, categorical_preprocessor

    def preprocess_num_cat(self, data):
        cat_col, encoder = self.cat_preprocess()
        data_categorical = data[cat_col]

        num_col, encoder_num = self.num_preprocess()
        data_numerical = data[num_col]

        data_encoded = encoder.fit_transform(data_categorical)
        data_encoded_num = encoder_num.fit_transform(data_numerical)

        print(f">> categorical encoded dataset contains {data_encoded.shape[1]} features")
        print(f">> numerical encoded dataset contains {data_encoded_num.shape[1]} features")

        columns_encoded = encoder.get_feature_names_out(data_categorical.columns)
        cat_rows = pd.DataFrame(data_encoded, columns=columns_encoded).astype(int)

        columns_encoded_num = encoder_num.get_feature_names_out(data_numerical.columns)
        num_rows = pd.DataFrame(data_encoded_num, columns=columns_encoded_num)

        data = pd.concat([num_rows, cat_rows], axis=1)
        return data

    def preprocess_cat(self, data):
        cat_col, encoder = self.cat_preprocess()
        data_categorical = data[cat_col]

        data_encoded = encoder.fit_transform(data_categorical)

        print(f">> encoded dataset contains {data_encoded.shape[1]} features")

        columns_encoded = encoder.get_feature_names_out(data_categorical.columns)
        cat_rows = pd.DataFrame(data_encoded, columns=columns_encoded).astype(int)

        data = data.drop(columns=cat_col)
        data = pd.concat([data, cat_rows], axis=1)
        return data
