#!/usr/bin/env python3

"""
@Author: Miro
@Date: 06/05/2022
@Version: 1.2
@Objective: creazione delle features
@TODO:
"""

import pandas as pd
from datetime import datetime
from configs import build_features_config as bfc, production_config as pc
from build_features_dir.data_features import DataFeatures
from build_features_dir.day_feature_build import DayFeatureBuild
from build_features_dir.operations_accounts_feature_build import OperationsAccountsFeatureBuild
from build_features_dir.evaluation_anagrafica_feature_build import EvaluationAnagraficaFeatureBuild


class BuildFeatures:
    def __init__(self, csv_data, max_elements=None, production=False,
                 subject_db=None, account_db=None, operations_db=None,
                 operations_subjects_db=None, target_db=None):

        start = datetime.now()

        self.data = DataFeatures(csv_data, max_elements, production,
                                 subject_db, account_db, operations_db,
                                 operations_subjects_db, target_db)

        print(">> input files loaded in ", datetime.now() - start)
        self.print_info()

    def print_info(self):
        print("\n>> GENERAL INFO")
        print(">> total evaluations loaded comportamenti", self.data.eval_subjects_comp.shape[0])
        print(">> total evaluations loaded day", self.data.eval_subjects_day.shape[0])
        print(">> total accounts file loaded ", self.data.accounts.shape[0])
        print(">> total operations loaded ", self.data.operations.shape[0])
        print(">> months considered ", self.data.months)
        print(">> prefix for each operations ", self.data.prefix_op_names)
        print(">> threshold variance 1 repetitiveness num ", self.data.variance_threshold_1)
        print(">> threshold variance 2 repetitiveness num ", self.data.variance_threshold_2)
        print(">> threshold variance filiale repetitiveness num ", self.data.variance_threshold_filiale)
        print(">> min age to consider for clustering ", self.data.min_age)
        print(">> max age to consider for clustering ", self.data.max_age)
        print(">> step age for clustering ", self.data.step_age)
        print(">> range repetitiveness [%] ", self.data.range_repetitiveness * 100)
        print(">> causal analytical grouping version ", self.data.causali_version)
        print(">> production ", self.data.production)

    def get_dataset_discovery_day(self, max_elements=None):
        if max_elements == 0 or self.data.max_elements_day == 0:
            return
        elif max_elements is None:
            self.data.max_elements = self.data.max_elements_day
        else:
            self.data.max_elements = max_elements

        start = datetime.now()
        print("\n>> DISCOVERY DAY AND ACCOUNTS FEATURES")
        DayFeatureBuild(self.data).build()
        print("   >> discovery day and accounts features done in ", datetime.now() - start)

    def get_dataset_discovery_comportamenti(self, max_elements=None):
        if max_elements == 0 or self.data.max_elements_comportamenti == 0:
            return
        elif max_elements is None:
            self.data.max_elements = self.data.max_elements_comportamenti
        else:
            self.data.max_elements = max_elements

        start = datetime.now()
        print("\n>> OPERATIONS AND ACCOUNTS FEATURES")
        OperationsAccountsFeatureBuild(self.data, self.data.eval_subjects_comp).build()
        print("   >> operations and accounts features done in ", datetime.now() - start)

        start = datetime.now()
        print("\n>> ANAGRAFICA AND EVALUATION FEATURES")
        EvaluationAnagraficaFeatureBuild(self.data, self.data.eval_subjects_comp).build()
        print("   >> anagrafica and evaluation features done in ", datetime.now() - start)

    def get_dataset_day_comportamenti(self, elements_day, elements_comp):
        self.get_dataset_discovery_comportamenti(max_elements=elements_comp)
        comportamenti_x, comportamenti_y = self.data.x, self.data.y
        self.data.x, self.data.y = pd.DataFrame(), pd.DataFrame()

        self.get_dataset_discovery_day(max_elements=elements_day)
        if self.data.production is False:
            self.data.y.replace(to_replace=0, value=2, inplace=True)
            self.data.y.replace(to_replace=1, value=3, inplace=True)

        if self.data.x.shape[0] > 0:
            day_col_x, comp_col_x = self.data.x.columns, None
            if comportamenti_x.shape[0] > 0:
                comp_col_x = comportamenti_x.columns
            else:
                comp_col_x = list(set(self.data.x.columns.values.tolist()) - set(bfc.day_col_x_only))
            self.fill_identificativi_vari(bfc.fill_with_string, comp_col_x, day_col_x)
        else:
            self.data.x = pd.concat([pd.DataFrame(columns=bfc.day_col_x_only), self.data.x], axis=1)
            bfc.day_col_x_only += comportamenti_x.columns.values.tolist()
            day_col_x = bfc.day_col_x_only

        self.data.x = pd.concat([comportamenti_x, self.data.x], ignore_index=True, sort=False)
        self.data.y = pd.concat([comportamenti_y, self.data.y], ignore_index=True, sort=False)

        if comportamenti_x.shape[0] > 0:
            self.fill_comportamenti_categories(comportamenti_x, day_col_x, bfc.fill_with_string)

    def fill_comportamenti_categories(self, comportamenti_x, day_col_x, fill_with_string):
        for col in self.data.x.columns:
            if (col in comportamenti_x.columns and col not in day_col_x) or \
                    (col in day_col_x and col not in comportamenti_x.columns):
                if col in fill_with_string:
                    self.data.x[col].fillna(bfc.value_default_dd_dc, inplace=True, limit=self.data.max_elements_comportamenti)
                else:
                    self.data.x[col].fillna(bfc.name_default_dd_dc, inplace=True, limit=self.data.max_elements_comportamenti)

    def fill_identificativi_vari(self, list_int, comp_col_x, day_col_x):
        for i, operation in enumerate(self.data.eval_subjects_day.A03.values.tolist()):
            if i >= self.data.max_elements:
                break
            if operation == 'Identificativi vari':
                self.check_cols_identification_vari(i, list_int, comp_col_x, day_col_x)

    def check_cols_identification_vari(self, i, list_int, comp_col_x, day_col_x):
        if self.data.production is False: self.data.y.at[i, 'EVALUATION'] -= 2
        for col in self.data.x.columns:
            if (col in comp_col_x and col not in day_col_x) or \
                    (col in day_col_x and col not in comp_col_x):
                if col in list_int: self.data.x.at[i, col] = bfc.value_default_dd_dc
                else: self.data.x.at[i, col] = bfc.name_default_dd_dc

    def get_dataset(self, discovery_day=False, discovery_comportamenti=False):
        elements_day, elements_comp = None, None
        if discovery_day is True and discovery_comportamenti is False:
            elements_comp, elements_day = 0, None
        elif discovery_day is False and discovery_comportamenti is True:
            elements_day, elements_comp = 0, None

        self.get_dataset_day_comportamenti(elements_day, elements_comp)
        self.data.x.ID = self.data.x.ID.astype(int)
        self.data.x.set_index(pc.index_name, inplace=True)

        if self.data.production is False: return self.data.x, self.data.y
        else: return self.data.x

    def extract_evaluation_found(self):
        rows_to_del = (self.data.y.index[self.data.y.EVALUATION == 0]).values.tolist()
        self.data.x_evaluated = self.data.x.drop(self.data.x.index[rows_to_del])
        return self.data.x_evaluated

    def save_dataset_csv(self):
        self.data.x.to_csv(self.data.path_x)
        print("\n>> dataset_x is saved in ", self.data.path_x)
        if self.data.production is False:
            self.data.y.to_csv(self.data.path_y, index=False)
            self.data.x_evaluated.to_csv(self.data.path_x_evaluated)
            print(">> dataset_y is saved in ", self.data.path_y)
            print(">> dataset x_evaluated is saved in ", self.data.path_x_evaluated)
