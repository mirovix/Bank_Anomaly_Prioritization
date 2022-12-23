#!/usr/bin/env python3

"""
@Author: Miro
@Date: 30/05/2022
@Version: 1.4
@Objective: caricamento dei file csv ottenuti dal database o dalle query
@TODO:
"""

import pandas as pd
import numpy as np
from configs import load_input_config as lic, production_config as pc
from build_operations_day_ds.operations_day_ds import build_day_df


class LoadData:
    def __init__(self):
        self.start_date_evaluation = lic.start_date_evaluation
        self.max_months_considered = lic.max_months_considered
        self.causal_analytical_csv = lic.causal_analytical_csv
        self.list_values_csv = lic.list_values_csv
        self.accounts_csv = lic.accounts_csv
        self.operations_csv = lic.operations_row_csv
        self.subject_csv = lic.subject_csv
        self.evaluation_csv = lic.evaluation_csv
        self.comment_csv = lic.comment_csv
        self.filename_csv = lic.filename_csv
        self.operations_subjects_csv = lic.operations_subjects_csv
        self.state_list_to_drop = lic.state_list_to_drop
        self.software_list_to_drop = lic.software_list_to_drop

        self.subjects = None
        self.operations = None
        self.operations_subjects = None

    def process_comments(self):
        comment = pd.read_csv(self.comment_csv, header=0, names=lic.cols_names_comment_evaluation_csv,
                              dtype=lic.dtypes_comm_file_csv).dropna()
        comment.sort_values(['ID', 'CREATION_DATE'], inplace=True)
        comment.BODY = comment.BODY.str.cat(comment[["CREATION_DATE"]].astype(str), sep=" >> ")
        comment.drop('CREATION_DATE', axis=1, inplace=True)
        return comment.groupby('ID', as_index=False).agg({'ID': 'first', 'BODY': ' >> NEXT COMMENT >> '.join})

    def process_filename(self):
        filename = pd.read_csv(self.filename_csv, header=0, names=lic.cols_names_file_evaluation_csv,
                               dtype=lic.dtypes_comm_file_csv)
        filename.sort_values(['ID', 'CREATION_DATE'], inplace=True)
        filename.drop('CREATION_DATE', axis=1, inplace=True)
        filename = filename.groupby('ID', as_index=False).agg({'ID': 'first', 'FILENAME': ' >> NEXT FILE >> '.join})
        filename.drop('FILENAME', axis=1, inplace=True)
        return filename.ID.values.tolist()

    def load_evaluation_not_processed(self, index_col=False, comment_col=False, file_name=False):
        if index_col is False:
            evaluation = pd.read_csv(self.evaluation_csv, low_memory=False, header=0,
                                     names=lic.cols_names_evaluation_csv, dtype=lic.dtypes_evaluation_csv)
        else:
            evaluation = pd.read_csv(self.evaluation_csv, low_memory=False, header=0,
                                     names=lic.cols_names_evaluation_csv, index_col=pc.index_name,
                                     dtype=lic.dtypes_evaluation_csv)
        if comment_col is True: evaluation = pd.merge(evaluation, self.process_comments(), how='left', on='ID')
        if file_name is True: evaluation.loc[evaluation.ID.isin(self.process_filename()), 'STATO'] = 'TO_ALERT'
        evaluation = evaluation.sort_values(lic.cols_rmv_duplicates)\
                               .drop_duplicates(lic.cols_rmv_duplicates, keep='last')\
                               .reset_index(drop=True)
        return evaluation

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

        target_info = target_info.sort_values(by=class_data_name).reset_index(drop=True)
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

    @staticmethod
    def cut_target(target_info):
        # target_info.drop(target_info[target_info.DATA < self.start_date_evaluation].index, inplace=True)
        # TO REMOVE 2022-05-06 01:08:52.003 >> 18 mesi
        # TO REMOVE 2022-08-31 00:00:00.000 >> 5 anni
        target_info.drop(target_info[target_info.DATA >= lic.end_date_evaluation].index, inplace=True)

    @staticmethod
    def clear_dataset_from_db(df):
        df.replace('NaT', np.nan, inplace=True)
        df.replace('None', np.nan, inplace=True)
        df.replace('NA', np.nan, inplace=True)
        df.fillna(value=np.nan, inplace=True)
        return df

    def load_evaluation(self, data=None, production_flag=False):
        if data is None:
            target_info = self.load_evaluation_not_processed()
            self.cut_target(target_info)
            target_info = self.process_input(target_info)
            target_info.IMPORTO = (target_info.IMPORTO.div(100)).map('{:.5f}'.format)
        else:
            production_flag = True
            target_info = data
            self.cut_target(target_info)
            self.clear_dataset_from_db(target_info)
            target_info = self.process_input(target_info)

        target_dd = self.process_target_day(target_info.copy(), production_flag)
        target_dc = self.process_target_comportamenti(target_info.copy())
        return target_dc, target_dd

    def load_subjects(self, data=None):

        if data is None:
            self.subjects = pd.read_csv(self.subject_csv, header=0,
                                        names=lic.cols_names_subject_csv,
                                        dtype=lic.dtypes_subject_csv)
        else:
            sql, engine = data
            self.subjects = pd.read_sql(sql, engine, columns=lic.cols_names_subject_csv).astype(lic.dtypes_subject_csv)

        self.subjects = self.clear_dataset_from_db(self.subjects)
        self.subjects.drop_duplicates(subset=['NDG'], ignore_index=True, inplace=True)
        self.subjects.GROSS_INCOME = self.subjects.GROSS_INCOME.replace(to_replace=np.nan, value=0.00000)
        self.subjects.ATECO = self.subjects.ATECO.replace(to_replace=' ', value=lic.ateco_to_replace)

        return self.subjects

    def load_row_operations(self, data=None):
        if data is None:
            self.operations = pd.read_csv(self.operations_csv, header=0,
                                          low_memory=False,
                                          names=lic.cols_names_operations_csv,
                                          dtype=str)
        else:
            sql_comp, engine_comp, sql_day, engine_day = data
            operations_info_comp = pd.read_sql(sql_comp, engine_comp, columns=lic.cols_names_operations_csv) \
                .astype(str)
            operations_info_day = pd.read_sql(sql_day, engine_day, columns=lic.cols_names_operations_csv) \
                .astype(str)

            self.operations = pd.concat([operations_info_comp, operations_info_day], ignore_index=True)
            self.operations = self.operations.sort_values("CONSOLIDATION_DATE", ascending=True) \
                .drop_duplicates(subset=["CODE_OPERATION"], keep="last")

        self.operations = self.clear_dataset_from_db(self.operations)

    def load_operations_processed(self):
        operations_processed = pd.merge(self.operations, self.operations_subjects, how='left', on='CODE_OPERATION')
        operations_processed.NDG = operations_processed.NDG.astype(np.int64)
        operations_processed = operations_processed[operations_processed.SUBJECT_TYPE == 'T']
        operations_processed.AMOUNT_CASH = operations_processed.AMOUNT_CASH.astype(np.float64).map('{:.2f}'.format)
        operations_processed.AMOUNT = operations_processed.AMOUNT.astype(np.float64).map('{:.2f}'.format)
        operations_processed.sort_values(["CODE_OPERATION", "NDG"], inplace=True)
        operations_processed.reset_index(drop=True, inplace=True)

        return operations_processed

    def load_accounts(self, data=None):
        if data is None:
            accounts = pd.read_csv(self.accounts_csv, header=0, names=lic.cols_names_accounts_csv,
                                   dtype=lic.dtypes_accounts_csv)
        else:
            sql, engine = data
            accounts = pd.read_sql(sql, engine, columns=lic.cols_names_accounts_csv).astype(lic.dtypes_accounts_csv)

        accounts.START_DATE = pd.to_datetime(accounts.START_DATE, format='%Y-%m-%d %H:%M:%S.%f',
                                             errors='coerce').astype(str)
        accounts.EXPIRE_DATE = pd.to_datetime(accounts.EXPIRE_DATE, format='%Y-%m-%d %H:%M:%S.%f',
                                              errors='coerce').astype(str)
        return self.clear_dataset_from_db(accounts)

    def load_list_values(self):
        return pd.read_csv(self.list_values_csv, header=0, sep=lic.sep,
                           names=lic.cols_names_list_values_csv, dtype=str,
                           keep_default_na=False, na_values=lic.default_list_values_csv)

    def load_causal_analytical(self):
        return pd.read_csv(self.causal_analytical_csv, dtype=str, header=0, sep=lic.sep)

    def load_operations_subjects(self, data=None):
        if data is None:
            self.operations_subjects = pd.read_csv(self.operations_subjects_csv,
                                                   names=lic.cols_names_operations_subjects_csv,
                                                   header=0, dtype=lic.dtypes_operations_subjects_csv)
        else:
            sql, engine = data

            self.operations_subjects = pd.read_sql(sql, engine, columns=lic.cols_names_operations_subjects_csv) \
                .astype(lic.dtypes_operations_subjects_csv)
        self.operations_subjects = self.clear_dataset_from_db(self.operations_subjects)
        return self.operations_subjects

    def load_operations_day(self):
        operations_day = build_day_df(self.operations, self.subjects, self.operations_subjects)

        operations_day.NDG = operations_day.NDG.astype(np.int64)
        operations_day.RISK_PROFILE_E = operations_day.RISK_PROFILE_E.astype(np.int8)

        operations_day.AMOUNT = operations_day.AMOUNT.astype(np.float64)
        operations_day.AMOUNT_CASH = operations_day.AMOUNT_CASH.astype(np.float64)

        operations_day.DATE_OPERATION = operations_day.DATE_OPERATION.astype(str)
        operations_day.DATE_OPERATION = pd.to_datetime(operations_day.DATE_OPERATION, format='%Y-%m-%d',
                                                       errors='coerce')
        operations_day.DATE_OPERATION = operations_day.DATE_OPERATION.astype(object)

        return operations_day
