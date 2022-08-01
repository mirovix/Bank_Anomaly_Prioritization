#!/usr/bin/env python3

"""
@Author: Miro
@Date: 09/06/2022
@Version: 1.0
@Objective: completamento delle features utilizzando datawig
@TODO:
"""

import pandas as pd
import datawig as dw
from load_dataset import load_dataset


class DatawigFilling:
    # def __init__(self, x_path="../../data/dataset/dataset_x.csv",
    #             path_to_corr_matrix="../../data/corr_matrix/corr_matrix.csv",
    #             path_to_save="../../data/dataset/ds_datawig/dataset_x_filled.csv",
    #             threshold=0.3):
    #    self.corr_matrix = pd.read_csv(path_to_corr_matrix)
    #    self.data = load_dataset(x_path, None, None).dropna()
    #    self.data_filled = load_dataset(x_path, None, None)
    #    self.path_to_save = path_to_save
    #    self.threshold = threshold
    #    self.splitting_values = [0.90, 0.10]
    #    self.other_to_del = ['DATA']

    def __init__(self, x_path="../data/dataset/dataset_x.csv",
                 path_to_save="../data/dataset/ds_datawig/dataset_x_filled.csv"):
        self.data = load_dataset(x_path, None, None)
        self.data_filled = None
        self.path_to_save = path_to_save

    def replace_values(self):
        for colm in self.data_filled.columns:
            for i in self.data_filled[colm][self.data_filled[colm].isnull()].index.values.tolist():
                value_filled = self.single_prediction(colm, self.data_filled.loc[[i]])
                self.data_filled[colm].loc[i] = [value_filled]

    def save(self):
        self.data_filled.to_csv(self.path_to_save, index=False)
        print(">> dataset_x_filled is saved in ", self.path_to_save)

    def build_simple_version(self, epochs=1000, prefix_name='./imputer_models/'):
        self.data_filled = dw.SimpleImputer.complete(self.data, num_epochs=epochs, verbose=0, output_path=prefix_name)
        self.save()

    def build(self, col_to_fill=None):
        all_predictions = []
        if col_to_fill is None:
            col_to_fill = self.corr_matrix

        for colm in col_to_fill:
            x_train, x_test = dw.utils.random_split(self.data, split_ratios=self.splitting_values)
            current_model = DatawigModel(x_train, x_test, colm, self.data_columns(colm), colm)
            predictions = current_model.train_datawig()
            all_predictions.append(predictions)
            # current_model.print_metrics()
        return all_predictions

    def data_columns(self, colm):
        cols_correlated_index = self.corr_matrix[colm][
            self.corr_matrix[colm] > self.threshold].index.values.tolist()
        name_cols_to_not_consider = self.corr_matrix.columns[cols_correlated_index].values.tolist()

        for e in self.other_to_del:
            name_cols_to_not_consider.append(e)

        output_col = [x for x in self.data.columns.values.tolist() if x not in name_cols_to_not_consider]
        return output_col

    @staticmethod
    def single_prediction(name, value):
        return DatawigModel(out_name=name).predict_datawig(value)


class DatawigModel:
    def __init__(self, x_train=None, x_test=None, out_name='', input_col=None, output_col='', epochs=20, batch_size=16,
                 prefix_name='./imputer_models/'):
        self.x_train = x_train
        self.x_test = x_test
        self.name = out_name
        self.input_col = input_col
        self.output_col = output_col
        self.epochs = epochs
        self.batch_size = batch_size
        self.prefix_name = prefix_name

    def print_metrics(self):
        impute = dw.SimpleImputer.load(self.prefix_name + self.name)
        metrics = impute.load_metrics()
        print("weighted_f1 ", metrics['weighted_f1'])
        print("avg_precision ", metrics['avg_precision'])

    def predict_datawig(self, value):
        impute = dw.SimpleImputer.load(self.prefix_name + self.name)
        predictions = impute.predict(value).values.tolist()
        return predictions[self.name + '_imputed'].values.tolist()[0]

    def train_datawig(self):
        impute = dw.SimpleImputer(
            input_columns=self.input_col,
            output_column=self.output_col,
            output_path=self.prefix_name + self.name
        )

        impute.fit(train_df=self.x_train, num_epochs=self.epochs, batch_size=self.batch_size)
        predictions = impute.predict(self.x_test)
        return predictions


DatawigFilling().build_simple_version()
