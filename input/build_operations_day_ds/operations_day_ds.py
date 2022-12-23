#!/usr/bin/env python3

"""
@Author: Miro
@Date: 18/11/2022
@Version: 1.0
@Objective: creazione del file csv tramite python, in alternativa è possibile utilizzare direttamente la query relativa al discovery day operations
@TODO:
"""

import numpy as np
import pandas as pd


def build_day_df(operations, subjects, operations_subjects):
    operations_cut = operations.copy()
    operations_cut = operations_cut[['CODE_OPERATION', 'DATE_OPERATION', 'CAUSAL',
                                     'SIGN', 'COUNTRY', 'AMOUNT', 'AMOUNT_CASH',
                                     'COUNTERPART_SUBJECT_COUNTRY']]

    subjects = subjects[['NDG', 'RESIDENCE_COUNTRY', 'RISK_PROFILE']]

    merge = pd.merge(operations_cut, operations_subjects, how='left', on='CODE_OPERATION')
    merge = merge[merge.SUBJECT_TYPE == 'T']
    merge = merge.drop(['SUBJECT_TYPE'], axis=1)

    operations_subject_executor = operations_subjects.copy()
    operations_subject_executor = operations_subject_executor[operations_subject_executor.SUBJECT_TYPE == 'E']
    operations_subject_executor.rename(columns={'NDG': 'NDG_E', 'SUBJECT_TYPE': 'SUBJECT_TYPE_E'}, inplace=True)

    merge = pd.merge(merge, operations_subject_executor, how='left', on='CODE_OPERATION')
    merge.drop(['SUBJECT_TYPE_E'], axis=1, inplace=True)
    merge.NDG_E = merge.NDG_E.fillna(merge.NDG).astype(np.int64)

    subjects_executor = subjects.copy()
    subjects_executor.rename(columns={'NDG': 'NDG_E',
                                      'RESIDENCE_COUNTRY': 'RESIDENCE_COUNTRY_E',
                                      'RISK_PROFILE': 'RISK_PROFILE_E'}, inplace=True)
    merge = pd.merge(merge, subjects_executor, how='left', on="NDG_E")
    merge.drop(['NDG_E'], axis=1, inplace=True)

    merge.sort_values(['CODE_OPERATION'], inplace=True)
    merge.reset_index(drop=True, inplace=True)
    return merge.astype(str)

