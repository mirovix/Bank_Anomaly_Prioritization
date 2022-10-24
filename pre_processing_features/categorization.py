#!/usr/bin/env python3

"""
@Author: Miro
@Date: 09/06/2022
@Version: 1.1
@Objective: categorizzazione delle features
@TODO:
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from configs import production_config
from fill_missing_values import FillMissingValues
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from configs import build_features_config as bfc, production_config as pc


class Categorization:
    def __init__(self, input_data=None):
        if input_data is not None:
            self.data = FillMissingValues(input_data, None).run_production()
            self.util()
        else:
            self.data, self.target, self.data_eval = load_dataset()
            self.data, self.target = FillMissingValues(self.data, self.target).run()
            self.data_anomaly_list = self.data.DATA.values.tolist()
            self.util()
            self.x_train, x_test_val, self.y_train, y_test_val = split_data(self.data, self.target, size_train=bfc.size_train)
            self.x_val, self.x_test, self.y_val, self.y_test = split_data(x_test_val, y_test_val, size_train=bfc.size_test)
            self.data_eval = self.data_eval.dropna()

    def util(self, col='DATA'):
        self.data = self.data.drop(columns=col)
        self.data[self.data.select_dtypes(np.float64).columns] = self.data.select_dtypes(np.float64).astype(
            np.float32)

    def run_production(self, path_encoders=production_config.machine_learning_categorization_data_path):
        encoder = joblib.load(path_encoders + bfc.data_enc_name)
        encoder_num = joblib.load(path_encoders + bfc.data_enc_num_name)
        columns_encoded = np.load(path_encoders + bfc.columns_encoded_name + '.npy', allow_pickle=True)
        columns_encoded_num = np.load(path_encoders + bfc.columns_encoded_num_name + '.npy', allow_pickle=True)
        cat_col = np.load(path_encoders + bfc.cat_col_name + '.npy', allow_pickle=True)
        num_col = np.load(path_encoders + bfc.num_col_name + '.npy', allow_pickle=True)

        try:
            cat_rows = pd.DataFrame(encoder.transform(self.data[cat_col]), columns=columns_encoded).astype(int)
            num_rows = pd.DataFrame(encoder_num.transform(self.data[num_col]), columns=columns_encoded_num)
            num_rows.insert(0, pc.index_name, self.data.index.values.tolist(), True)
            result = pd.concat([num_rows, cat_rows], axis=1)
            result.set_index(pc.index_name, inplace=True)
        except IOError:
            return pd.DataFrame()
        return result

    def run(self):
        data = (self.x_train, self.x_val, self.x_test)
        return (self.preprocess_num_cat_train(data)), (self.y_train, self.y_val, self.y_test)

    @staticmethod
    def preprocess_num_cat_train(data, path_to_save=production_config.machine_learning_categorization_data_path):
        x_train, x_val, x_test = data

        cat_col, encoder = cat_preprocess(x_train)
        data_categorical = x_train[cat_col]

        num_col, encoder_num = num_preprocess(x_train)
        data_numerical = x_train[num_col]

        data_encoded = encoder.fit(data_categorical)
        data_encoded_num = encoder_num.fit(data_numerical)

        columns_encoded = encoder.get_feature_names_out(data_categorical.columns)
        columns_encoded_num = encoder_num.get_feature_names_out(data_numerical.columns)

        np.save(path_to_save + bfc.columns_encoded_name, columns_encoded)
        np.save(path_to_save + bfc.columns_encoded_num_name, columns_encoded_num)
        np.save(path_to_save + bfc.cat_col_name, cat_col)
        np.save(path_to_save + bfc.num_col_name, num_col)
        joblib.dump(data_encoded, path_to_save + bfc.data_enc_name)
        joblib.dump(data_encoded_num, path_to_save + bfc.data_enc_num_name)

        result = []
        for i, x in enumerate(data):
            cat_rows = pd.DataFrame(data_encoded.transform(x[cat_col]), columns=columns_encoded).astype(int)
            num_rows = pd.DataFrame(data_encoded_num.transform(x[num_col]), columns=columns_encoded_num)
            num_rows.insert(0, pc.index_name, x.index.values.tolist(), True)
            result.append(pd.concat([num_rows, cat_rows], axis=1))
            result[i].set_index(pc.index_name, inplace=True)

        return result[0], result[1], result[2]


def num_preprocess(data):
    numerical_columns_selector = selector(dtype_exclude=object)
    numerical_columns = numerical_columns_selector(data)
    numerical_preprocessor = MinMaxScaler()
    return numerical_columns, numerical_preprocessor


def cat_preprocess(data):
    categorical_columns_selector = selector(dtype_include=object)
    categorical_columns = categorical_columns_selector(data)
    categorical_preprocessor = OneHotEncoder(sparse=False, handle_unknown='ignore')
    return categorical_columns, categorical_preprocessor


def load_dataset(flag_y=True, flag_x_eval=True):
    x_dataset = pd.read_csv(bfc.path_x, low_memory=False, index_col=[pc.index_name])
    print("\n>> dataset_x is loaded from ", bfc.path_x)
    if flag_y is False and flag_x_eval is False:
        return x_dataset

    y_dataset = pd.read_csv(bfc.path_y)
    print(">> dataset_y is loaded from ", bfc.path_y)
    if flag_x_eval is False:
        return x_dataset, y_dataset

    x_eval_dataset = pd.read_csv(bfc.path_x_evaluated, low_memory=False)
    print(">> dataset x_evaluated is loaded from ", bfc.path_x_evaluated)

    return x_dataset, y_dataset, x_eval_dataset


def split_data(data, target, size_train):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=(1 - size_train), shuffle=True)
    return x_train, x_test, y_train, y_test
