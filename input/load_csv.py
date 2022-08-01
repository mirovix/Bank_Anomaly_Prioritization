#!/usr/bin/env python3

"""
@Author: Miro
@Date: 30/05/2022
@Version: 1.2
@Objective: caricamento dei file csv ottenuti dal database
@TODO:
"""

import pandas as pd
import numpy as np
from build_features import range_date


class LoadData:
    def __init__(self, evaluation_csv=r'../data/target_not_processed.csv',
                 subject_csv=r'../data/all_subjects_db.csv',
                 operations_csv=r'../data/all_operations_db.csv',
                 accounts_csv=r'../data/all_accounts_db.csv',
                 list_values_csv=r'../data/list_values.csv',
                 causal_analytical_csv=r'../data/causale_analitica_V2.csv',
                 operations_day_csv=r'../data/all_operations_day_db.csv',
                 start_date_evaluation='2020-07-01 00:00:00.001',
                 max_months_considered=19,
                 software_list_to_drop=None,
                 state_list_to_drop=None
                 ):

        if state_list_to_drop is None:
            state_list_to_drop = ['NOT_EVALUATE', 'NOT_TO_ALERT_A', 'VALUATING']
        self.state_list_to_drop = state_list_to_drop
        if software_list_to_drop is None:
            software_list_to_drop = ['EXT_SYS', 'USURA', 'UIFCOM']
        self.software_list_to_drop = software_list_to_drop
        self.start_date_evaluation = start_date_evaluation
        self.max_months_considered = max_months_considered
        self.causal_analytical_csv = causal_analytical_csv
        self.list_values_csv = list_values_csv
        self.accounts_csv = accounts_csv
        self.operations_csv = operations_csv
        self.operations_day_csv = operations_day_csv
        self.subject_csv = subject_csv
        self.evaluation_csv = evaluation_csv

    def load_evaluation(self):

        target_info = pd.read_csv(self.evaluation_csv, header=0, sep=';')

        target_info.IMPORTO = target_info.IMPORTO.replace(to_replace='Vari', value=np.nan)
        target_info.IMPORTO = target_info.IMPORTO.astype(np.float64).map('{:.2f}'.format)

        # target_info.NDG = target_info.NDG.replace(to_replace=np.nan, value='-1')
        # target_info.drop(target_info[target_info.NDG == int(-1)].index, inplace=True)

        target_info.dropna(inplace=True)
        target_info.NDG = target_info.NDG.astype(np.int64)

        drop_stato = False
        for state_to_drop in self.state_list_to_drop:
            drop_stato |= (target_info.STATO == state_to_drop)
        target_info.drop(target_info[drop_stato].index, inplace=True)
        target_info.STATO = target_info.STATO.replace({'NOT_TO_ALERT': 0, 'TO_ALERT': 1})

        drop_software = False
        for software_to_drop in self.software_list_to_drop:
            drop_software |= (target_info.SOFTWARE == software_to_drop)
        target_info.drop(target_info[drop_software].index, inplace=True)

        drop_old_eval = target_info.DATA < self.start_date_evaluation
        target_info.drop(target_info[drop_old_eval].index, inplace=True)

        target_info = target_info.sort_values(by="DATA")

        target_info_discovery_day = target_info.copy()
        target_info_discovery_day.drop(target_info_discovery_day[target_info_discovery_day.SOFTWARE == 'COMPORTAMENT'].index, inplace=True)
        target_info_discovery_day.drop(target_info_discovery_day[target_info_discovery_day.STATO == 'VALUATING_AAU'].index, inplace=True)

        target_info_discovery_comportamenti = target_info.copy()
        target_info_discovery_comportamenti.drop(target_info_discovery_comportamenti[target_info_discovery_comportamenti.SOFTWARE == 'DISCOVERY'].index, inplace=True)

        return target_info_discovery_comportamenti, target_info_discovery_day

    def load_subjects(self):
        cols_names = ['NDG', 'BIRTH_DAY', 'LEGAL_SPECIE', 'RESIDENCE_CAB', 'RESIDENCE_CAP', 'RESIDENCE_PROVINCE',
                      'RESIDENCE_CITY', 'RESIDENCE_COUNTRY', 'SAE', 'ATECO', 'SSE', 'RISK_PROFILE',
                      'FIRST_CONTACT_DATE', 'STATUS', 'REPORTED', 'PREJUDICIAL', 'NCHECKREQUIRED', 'NCHECKDEBITED',
                      'NCHECKAVAILABLE', 'PORTFOLIO', 'INSERT_DATE', 'LAST_UPDATE_DATE', 'GROSS_INCOME',
                      'SETTLEMENT_DATE']

        dtypes = {'NDG': np.int64, "REPORTED": pd.Int64Dtype(), "NCHECKREQUIRED": pd.Int64Dtype(),
                  'NCHECKDEBITED': pd.Int64Dtype(), 'NCHECKAVAILABLE': pd.Int64Dtype(),
                  'RISK_PROFILE': pd.Int64Dtype(), 'BIRTH_DAY': str, 'RESIDENCE_CAP': str, 'RESIDENCE_COUNTRY': str,
                  'SAE': str, 'ATECO': str, 'LEGAL_SPECIE': str, 'GROSS_INCOME': np.float64}

        subjects_info = pd.read_csv(self.subject_csv, header=0, sep=';', names=cols_names, dtype=dtypes)

        subjects_info.ATECO = subjects_info.ATECO.replace(to_replace=' ', value=-1)

        return subjects_info

    def load_operations(self):
        cols_names = ['NDG', 'CODE_OPERATION', 'CODE_FLUX', 'ACCOUNT', 'ACCOUNT_SUBTYPE', 'ACCOUNT_RELATION',
                      'DATE_OPERATION', 'CAUSAL', 'SUBCAUSAL', 'SIGN', 'FLAG_FRACTION', 'FLAG_CASH', 'AMOUNT',
                      'AMOUNT_CASH', 'COUNTRY', 'CURRENCY', 'CURRENCY_TYPE', 'CONSOLIDATION_DATE',
                      'TRANSACTION_CODE', 'TRANSACTION_TYPE', 'COUNTERPART_TYPE', 'COUNTERPART_CODE', 'COUNTERPART_CAB',
                      'COUNTERPART_PROVINCE', 'COUNTERPART_CITY', 'COUNTERPART_COUNTRY', 'COUNTERPART_ACCOUNT',
                      'COUNTERPART_SUBJECT_COUNTRY', 'FILIAL']

        dtypes = {'NDG': np.int64}
        for i in range(1, len(cols_names)):
            dtypes[cols_names[i]] = str

        operations_info = pd.read_csv(self.operations_csv, header=0, sep=';', names=cols_names, dtype=dtypes)

        operations_info.AMOUNT = operations_info.AMOUNT.astype(np.float64).map('{:.2f}'.format)
        # start_range, end_range = range_date(self.start_date_evaluation, self.max_months_considered, months_to_remove=0)
        # query_conditions = (operations_info.DATE_OPERATION < str(start_range)) & (
        #            operations_info.DATE_OPERATION > str(end_range))

        return operations_info  # [query_conditions]

    def load_accounts(self):
        cols_names = ['CODE_ACCOUNT', 'START_DATE', 'EXPIRE_DATE', 'NDG']
        dtypes = {cols_names[0]: str, cols_names[1]: str, cols_names[2]: str, cols_names[3]: np.int64}
        return pd.read_csv(self.accounts_csv, header=0, sep=';', names=cols_names, dtype=dtypes)

    def load_list_values(self):
        cols_names = ['RISCHIO_PAESE_ALTISSIMO', 'RISCHIO_PAESE_ALTO', 'PAESI_RESIDENZA', 'ATECO_0', 'ATECO_1',
                      'ATECO_2', 'ATECO_3', 'ATECO_SAE_NONE', 'SAE_0', 'SAE_1', 'SAE_2', 'SAE_3', 'SAE_4', 'PRV_0',
                      'PRV_1', 'PRV_2', 'PRV_3', 'PRV_4']
        dtypes = {}
        for i in range(len(cols_names)):
            dtypes[cols_names[i]] = str
        return pd.read_csv(self.list_values_csv, header=0, sep=';', names=cols_names, dtype=dtypes)

    def load_causal_analytical(self):
        return pd.read_csv(self.causal_analytical_csv, dtype=str, header=0, sep=';')

    def load_operations_day(self):
        operations_day = pd.read_csv(self.operations_day_csv, dtype=str, header=0, sep=';')
        operations_day.NDG = operations_day.NDG.astype(np.int64)
        operations_day.RISK_PROFILE_E = operations_day.RISK_PROFILE_E.astype(np.int8)
        operations_day.AMOUNT = operations_day.AMOUNT.astype(np.float64)
        operations_day.AMOUNT_CASH = operations_day.AMOUNT_CASH.astype(np.float64)
        return operations_day
