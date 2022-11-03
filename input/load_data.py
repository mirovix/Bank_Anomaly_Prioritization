#!/usr/bin/env python3

"""
@Author: Miro
@Date: 30/05/2022
@Version: 1.3
@Objective: caricamento dei file csv ottenuti dal database o dalle query
@TODO:
"""

import pandas as pd
import numpy as np
from configs import load_input_config as lic, production_config as pc


class LoadData:
    def __init__(self):
        self.start_date_evaluation = lic.start_date_evaluation
        self.max_months_considered = lic.max_months_considered
        self.causal_analytical_csv = lic.causal_analytical_csv
        self.list_values_csv = lic.list_values_csv
        self.accounts_csv = lic.accounts_csv
        self.operations_csv = lic.operations_csv
        self.operations_day_csv = lic.operations_day_csv
        self.subject_csv = lic.subject_csv
        self.evaluation_csv = lic.evaluation_csv
        self.state_list_to_drop = lic.state_list_to_drop
        self.software_list_to_drop = lic.software_list_to_drop

    def load_evaluation_not_processed(self, index_col=False):
        if index_col is False:
            return pd.read_csv(self.evaluation_csv, low_memory=False, header=0,
                               names=lic.cols_names_evaluation_csv, dtype=lic.dtypes_evaluation_csv)
        else:
            return pd.read_csv(self.evaluation_csv, low_memory=False, header=0,
                               names=lic.cols_names_evaluation_csv, index_col=pc.index_name,
                               dtype=lic.dtypes_evaluation_csv)

    def process_input(self, target_info, class_data_name='DATA'):
        target_info.IMPORTO = target_info.IMPORTO.replace(to_replace='Vari', value=lic.importo_to_replace)
        target_info.IMPORTO = target_info.IMPORTO.astype(np.float64)

        drop_stato = False
        for state_to_drop in self.state_list_to_drop:
            drop_stato |= (target_info.STATO == state_to_drop)
        target_info.drop(target_info[drop_stato].index, inplace=True)

        drop_software = False
        for software_to_drop in self.software_list_to_drop:
            drop_software |= (target_info.SOFTWARE == software_to_drop)
        target_info.drop(target_info[drop_software].index, inplace=True)

        target_info = target_info.sort_values(by=class_data_name)
        return target_info

    @staticmethod
    def process_target_day(target_dd, production_flag):
        target_dd.drop(target_dd[target_dd.SOFTWARE.replace(' ', '') == 'COMPORTAMENT'].index, inplace=True)
        if target_dd.shape[0] == 0:
            return target_dd

        target_dd.dropna(inplace=True)
        target_dd.NDG = target_dd.NDG.astype(np.int64)
        target_dd.STATO.replace({'NOT_TO_ALERT': 0, 'TO_ALERT': 1}, inplace=True)

        if production_flag is False: target_dd.DATA_OPERATION = pd.to_datetime(
            target_dd.DATA_OPERATION, format='%Y%m%d', errors='coerce')

        target_dd.DATA_OPERATION = target_dd.DATA_OPERATION.astype(object)
        return target_dd

    @staticmethod
    def process_target_comportamenti(target_dc):
        target_dc.drop(target_dc[target_dc.SOFTWARE.replace(' ', '') == 'DISCOVERY'].index, inplace=True)
        if target_dc.shape[0] == 0:
            return target_dc

        target_dc.drop('DATA_OPERATION', inplace=True, axis=1, errors='ignore')
        target_dc.drop('A03', inplace=True, axis=1, errors='ignore')
        target_dc.dropna(inplace=True)
        target_dc.NDG = target_dc.NDG.astype(np.int64)
        target_dc.STATO.replace({'NOT_TO_ALERT': 0, 'TO_ALERT': 1}, inplace=True)

        return target_dc

    def load_evaluation(self, data=None, production_flag=False):
        if data is None:
            target_info = self.load_evaluation_not_processed()
            target_info = self.process_input(target_info)
            target_info.drop(target_info[target_info.DATA < self.start_date_evaluation].index, inplace=True)
            # TO REMOVE 2022-05-06 01:08:52.003
            target_info.drop(target_info[target_info.DATA >= lic.end_date_evaluation].index, inplace=True)
            # target_info.drop(target_info[target_info.STATO == 'VALUATING_AAU'].index, inplace=True)
            target_info.IMPORTO = (target_info.IMPORTO.div(100)).map('{:.2f}'.format)
        else:
            production_flag = True
            target_info = data
            target_info = self.process_input(target_info)

        target_dd = self.process_target_day(target_info.copy(), production_flag)
        target_dc = self.process_target_comportamenti(target_info.copy())
        return target_dc, target_dd

    def load_subjects(self, data=None):

        if data is None:
            subjects_info = pd.read_csv(self.subject_csv, header=0, sep=lic.sep,
                                        names=lic.cols_names_subject_csv, dtype=lic.dtypes_subject_csv)
        else:
            sql, engine = data
            subjects_info = pd.read_sql(sql, engine).astype(lic.dtypes_subject_csv)
            subjects_info.replace('NaT', np.nan, inplace=True)
            subjects_info.replace('None', np.nan, inplace=True)

        subjects_info.ATECO = subjects_info.ATECO.replace(to_replace=' ', value=lic.ateco_to_replace)

        return subjects_info

    def load_operations(self, data=None):
        if data is None:
            operations_info = pd.read_csv(self.operations_csv, header=0,
                                          names=lic.cols_names_operations_csv,
                                          dtype=lic.dtypes_operations_csv, sep=lic.sep)
        else:
            sql_comp, engine_comp, sql_day, engine_day = data
            operations_info_comp, operations_info_day = None, None
            if sql_comp is not None:
                operations_info_comp = pd.read_sql(sql_comp, engine_comp).astype(lic.dtypes_operations_csv)
            if sql_day is not None:
                operations_info_day = pd.read_sql(sql_day, engine_day).astype(lic.dtypes_operations_csv)
            operations_info = pd.concat([operations_info_comp, operations_info_day], ignore_index=True) \
                              .sort_values("CONSOLIDATION_DATE", ascending=True) \
                              .drop_duplicates(subset=["CODE_OPERATION"], keep="last")
            operations_info.replace('NaT', np.nan, inplace=True)
            operations_info.replace('None', np.nan, inplace=True)

        operations_info.AMOUNT = operations_info.AMOUNT.astype(np.float64).map('{:.2f}'.format)

        return operations_info

    def load_accounts(self, data=None):
        if data is None:
            return pd.read_csv(self.accounts_csv, header=0, sep=lic.sep, names=lic.cols_names_accounts_csv, dtype=lic.dtypes_accounts_csv)
        else:
            sql, engine = data
            accounts = pd.read_sql(sql, engine).astype(lic.dtypes_accounts_csv)
            accounts.replace('NaT', np.nan, inplace=True)
            accounts.replace('None', np.nan, inplace=True)
            return accounts

    def load_list_values(self):
        return pd.read_csv(self.list_values_csv, header=0, sep=lic.sep,
                           names=lic.cols_names_list_values_csv, dtype=str,
                           keep_default_na=False, na_values=lic.default_list_values_csv)

    def load_causal_analytical(self):
        return pd.read_csv(self.causal_analytical_csv, dtype=str, header=0, sep=lic.sep)

    def load_operations_day(self, data=None):
        if data is None:
            operations_day = pd.read_csv(self.operations_day_csv, dtype=str, header=0,
                                         names=lic.cols_names_operations_day_csv, sep=lic.sep)
        else:
            sql, engine = data
            operations_day = pd.read_sql(sql, engine).astype(str)
            operations_day.replace('NaT', np.nan, inplace=True)
            operations_day.replace('None', np.nan, inplace=True)

        operations_day.NDG = operations_day.NDG.astype(np.int64)
        operations_day.RISK_PROFILE_E = operations_day.RISK_PROFILE_E.astype(np.int8)
        operations_day.AMOUNT = operations_day.AMOUNT.astype(np.float64)
        operations_day.AMOUNT_CASH = operations_day.AMOUNT_CASH.astype(np.float64)
        operations_day.DATE_OPERATION = operations_day.DATE_OPERATION.astype(str)
        operations_day.DATE_OPERATION = pd.to_datetime(operations_day.DATE_OPERATION, format='%Y-%m-%d', errors='coerce')
        operations_day.DATE_OPERATION = operations_day.DATE_OPERATION.astype(object)
        return operations_day
