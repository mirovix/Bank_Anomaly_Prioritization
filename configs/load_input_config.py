"""
@Author: Miro
@Date: 26/10/2022
@Version: 1.0
@Objective: configuration file for loading input files
@TODO:
"""

import numpy as np
import pandas as pd
from configs import production_config as pc

# general input data for training

evaluation_csv = pc.base_path + '/data/row_data/target_not_processed.csv'
comment_csv = pc.base_path + '/data/row_data/target_comment.csv'
filename_csv = pc.base_path + '/data/row_data/target_file_name.csv'

subject_csv = pc.base_path + '/data/row_data/all_subjects_db.csv'
operations_row_csv = pc.base_path + '/data/row_data/all_operations_db_row.csv'
accounts_csv = pc.base_path + '/data/row_data/all_accounts_db.csv'
list_values_csv = pc.base_path + '/data/row_data/list_values.csv'
causal_analytical_csv = pc.base_path + '/data/row_data/causale_analitica_V2.csv'
operations_subjects_csv = pc.base_path + '/data/row_data/all_operations_subjects_db.csv'
start_date_evaluation = '2020-07-01 00:00:00.001'
end_date_evaluation = '2022-08-31 00:00:00.001'
max_months_considered = 19
software_list_to_drop = ['EXT_SYS', 'USURA', 'UIFCOM']
state_list_to_drop = ['NOT_EVALUATE', 'NOT_TO_ALERT_A', 'VALUATING', 'VALUATING_AAU']
sep = ';'

# evaluations data

cols_names_evaluation_csv = ['ID', 'CODICE_ANOMALIA', 'SOFTWARE', 'IMPORTO', 'DATA', 'STATO', 'NDG', 'DATA_OPERATION', 'A03']
cols_rmv_duplicates = ['CODICE_ANOMALIA', 'SOFTWARE', 'IMPORTO', 'STATO', 'NDG', 'DATA_OPERATION', 'A03']
cols_names_comment_evaluation_csv = ['ID', 'CREATION_DATE', 'BODY']
cols_names_file_evaluation_csv = ['ID', 'CREATION_DATE', 'FILENAME']
dtypes_evaluation_csv = {'NDG': str, 'DATA_OPERATION': str}
dtypes_comm_file_csv = {'CREATION_DATE': str}
importo_to_replace = -10000000

# subjects data

cols_names_subject_csv = ['NDG', 'BIRTH_DAY', 'LEGAL_SPECIE', 'RESIDENCE_CAB', 'RESIDENCE_CAP', 'RESIDENCE_PROVINCE',
                          'RESIDENCE_CITY', 'RESIDENCE_COUNTRY', 'SAE', 'ATECO', 'SSE', 'RISK_PROFILE',
                          'FIRST_CONTACT_DATE', 'STATUS', 'REPORTED', 'PREJUDICIAL', 'NCHECKREQUIRED', 'NCHECKDEBITED',
                          'NCHECKAVAILABLE', 'PORTFOLIO', 'INSERT_DATE', 'LAST_UPDATE_DATE', 'GROSS_INCOME',
                          'GROSS_INCOME_WAGE',
                          'SETTLEMENT_DATE']
dtypes_subject_csv = {'NDG': np.int64, "REPORTED": pd.Int64Dtype(), "NCHECKREQUIRED": pd.Int64Dtype(),
                      'NCHECKDEBITED': pd.Int64Dtype(), 'NCHECKAVAILABLE': pd.Int64Dtype(),
                      'RISK_PROFILE': pd.Int64Dtype(), 'BIRTH_DAY': str, 'RESIDENCE_CAP': str, 'RESIDENCE_COUNTRY': str,
                      'SAE': str, 'ATECO': str, 'LEGAL_SPECIE': str, 'GROSS_INCOME': np.float64,
                      'GROSS_INCOME_WAGE': np.float64}
ateco_to_replace = -1

# operations data

cols_names_operations_csv = ['CODE_OPERATION', 'ACCOUNT', 'ACCOUNT_SUBTYPE', 'ACCOUNT_RELATION', 'DATE_OPERATION',
                             'CAUSAL', 'SUBCAUSAL', 'SIGN', 'FLAG_FRACTION', 'FLAG_CASH', 'AMOUNT', 'AMOUNT_CASH',
                             'COUNTRY', 'CURRENCY', 'CURRENCY_TYPE', 'CONSOLIDATION_DATE', 'TRANSACTION_CODE',
                             'TRANSACTION_TYPE', 'COUNTERPART_TYPE', 'COUNTERPART_CODE', 'COUNTERPART_CAB',
                             'COUNTERPART_PROVINCE', 'COUNTERPART_CITY', 'COUNTERPART_COUNTRY', 'COUNTERPART_ACCOUNT',
                             'COUNTERPART_SUBJECT_COUNTRY', 'FILIAL']

# operations_subjects data

cols_names_operations_subjects_csv = ['CODE_OPERATION', 'NDG', 'SUBJECT_TYPE']
dtypes_operations_subjects_csv = {cols_names_operations_subjects_csv[0]: str,
                                  cols_names_operations_subjects_csv[1]: np.int64,
                                  cols_names_operations_subjects_csv[2]: str}

# accounts data

cols_names_accounts_csv = ['CODE_ACCOUNT', 'START_DATE', 'EXPIRE_DATE', 'NDG']
dtypes_accounts_csv = {cols_names_accounts_csv[0]: str, cols_names_accounts_csv[1]: str,
                       cols_names_accounts_csv[2]: str, cols_names_accounts_csv[3]: np.int64}

# data for province, ateco and sae categorization

cols_names_list_values_csv = ['RISCHIO_PAESE_ALTISSIMO', 'RISCHIO_PAESE_ALTO', 'PAESI_RESIDENZA', 'ATECO_0', 'ATECO_1',
                              'ATECO_2', 'ATECO_3', 'ATECO_SAE_NONE', 'SAE_0', 'SAE_1', 'SAE_2', 'SAE_3', 'SAE_4',
                              'PRV_0', 'PRV_1', 'PRV_2', 'PRV_3', 'PRV_4']
default_list_values_csv = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND',
                           '-1.#QNAN', '-NaN', '-nan', '1.#IND',
                           '1.#QNAN', 'N/A', 'NULL', 'NaN',
                           'n/a', 'nan', 'null']

