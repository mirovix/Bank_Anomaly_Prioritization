#!/usr/bin/env python3

"""
@Author: Miro
@Date: 20/10/2022
@Version: 1.0
@Objective: costruzione delle features legate alle operazioni e agli account
@TODO:
"""

import pandas as pd
import numpy as np
import sys
from dateutil.relativedelta import relativedelta
from data_features import range_date
from build_features_dir.operations_accounts_feature_definition import OperationsAccountsFeatureDefinition as oafs


class OperationsAccountsFeatureBuild:
    def __init__(self, data, evaluation_subjects_data):
        self.data = data
        self.evaluation_subjects_data = evaluation_subjects_data
        self.operations_accounts_feature_data = oafs(self.data).operations_accounts_feature()
        self.op_features = None
        self.current_query_table = None
        self.current_total_amount = 0

    def build(self):
        self.insert_table_operation_account(self.operations_accounts_feature_data)

    def creation_table_query_evaluations(self, query_prop):
        data_anomaly, num_months, ndg = query_prop
        start_range, end_range = range_date(data_anomaly, num_months)
        query_conditions = (self.evaluation_subjects_data.DATA < str(start_range)) & \
                           (self.evaluation_subjects_data.DATA > str(end_range)) & \
                           (self.evaluation_subjects_data.NDG == ndg)
        return self.evaluation_subjects_data[query_conditions]

    def creation_table_query_operations(self, query_prop, table):
        sign, casual_condition_list = query_prop
        casual_condition = self.multiple_causal_analytical(table, casual_condition_list)
        query_conditions = (table.SIGN == sign) & casual_condition
        return table[query_conditions]

    @staticmethod
    def creation_initial_query_table_ndg(table, date, max_months, ndg_condition, type_operations_account=1):
        start_range, end_range = range_date(date, max_months)

        if type_operations_account == 1: date = table.DATE_OPERATION
        else: date = table.START_DATE

        query_conditions = (date < str(start_range)) & (date > str(end_range)) & ndg_condition
        return table[query_conditions]

    def find_repetitiveness_movimenti(self, query_table):
        max_rip = 0
        list_amount = (query_table.AMOUNT.astype(np.float64)).values.tolist()
        for i in range(query_table.shape[0] - 1):
            current_rip = 0
            for j in range(i + 1, query_table.shape[0]):
                range_amount = list_amount[i] * self.data.range_repetitiveness
                if (list_amount[i] - range_amount) <= list_amount[j] <= (list_amount[i] + range_amount):
                    current_rip += 1

            if max_rip < current_rip: max_rip = current_rip + 1
        return max_rip

    @staticmethod
    def mean_list_operation_per_value(query_table, data_anomaly, months):
        rip_elements = []
        for m in range(months):
            start_range, end_range = range_date(data_anomaly, (m + 1))
            start_range -= relativedelta(months=m)
            query_conditions = (query_table.DATE_OPERATION < str(start_range)) & (query_table.DATE_OPERATION > str(end_range))
            rip_elements.append(query_table[query_conditions].shape[0])

        media_soglia1 = sum(rip_elements) / months
        return rip_elements, media_soglia1

    @staticmethod
    def find_repetitiveness_num(rip_elements, media_soglia1, th1, th2):
        rip_elements_processed = []
        max_rip, media_soglia2 = 0, 0
        if len(rip_elements) == 0: return 0

        for e in rip_elements:
            soglia = e * th1
            if abs(media_soglia1 - e) ** 2 < soglia and e > 0:
                rip_elements_processed.append(e)

        if len(rip_elements_processed) == 0: return 0

        media_soglia2 = sum(rip_elements_processed) / len(rip_elements_processed)
        soglia = media_soglia2 * th2
        for e in rip_elements_processed:
            if abs(media_soglia2 - e) ** 2 < soglia:
                max_rip += 1

        if max_rip < 2: max_rip = 0

        return max_rip

    def insert_table_operation_account(self, all_tables_information):
        self.op_features = [[] for _ in range(len(all_tables_information))]
        for i, evaluation in self.evaluation_subjects_data.iterrows():

            if i >= self.data.max_elements: break

            to_write = "   >> evaluations completed " + str(i + 1) + "/" + str(self.data.max_elements) + "\r"
            sys.stdout.write(to_write)

            single_evaluation_table_6m = self.creation_initial_query_table_ndg(
                table=self.data.operations, date=evaluation.DATA, max_months=self.data.months[1],
                ndg_condition=(self.data.operations.NDG == evaluation.NDG), type_operations_account=1)
            single_evaluation_table_3m = self.creation_initial_query_table_ndg(
                table=single_evaluation_table_6m, date=evaluation.DATA, max_months=self.data.months[0], ndg_condition=1,
                type_operations_account=1)
            single_evaluation_table_6m_account = self.creation_initial_query_table_ndg(
                table=self.data.accounts, date=evaluation.DATA, max_months=self.data.months[1],
                ndg_condition=(self.data.accounts.NDG == evaluation.NDG), type_operations_account=0)
            single_evaluation_table_3m_account = self.creation_initial_query_table_ndg(
                table=single_evaluation_table_6m_account, date=evaluation.DATA, max_months=self.data.months[0],
                ndg_condition=1, type_operations_account=0)

            for j, single_table_info in enumerate(all_tables_information):
                check_operations_account = single_table_info[5]
                if check_operations_account == 0:
                    self.insert_operation_evaluation(j, single_table_info, evaluation, single_evaluation_table_3m,
                                                     single_evaluation_table_6m)
                else:
                    self.insert_account_evaluation(j, single_table_info, single_evaluation_table_3m_account,
                                                   single_evaluation_table_6m_account)

        for k, op_f in enumerate(self.op_features):
            temp_df = pd.DataFrame(op_f, columns=[all_tables_information[k][0].upper()])
            self.data.x = pd.concat([self.data.x, temp_df], axis=1)

    def insert_account_evaluation(self, j, single_table_info, single_evaluation_table_3m_account,
                                  single_evaluation_table_6m_account):
        month, type_account_feature = single_table_info[1], single_table_info[2]
        if month == self.data.months[0]:
            self.op_features[j].append(self.accounts_feature(single_evaluation_table_3m_account, type_account_feature))
        else:
            self.op_features[j].append(self.accounts_feature(single_evaluation_table_6m_account, type_account_feature))

    def insert_operation_evaluation(self, j, single_table_info, evaluation, single_evaluation_table_3m,
                                    single_evaluation_table_6m):
        check_build_new_table_and_type = single_table_info[4]
        date = evaluation.DATA
        month = single_table_info[2]
        ndg = evaluation.NDG
        if check_build_new_table_and_type == 0 or check_build_new_table_and_type == 6:
            sign, casual_condition_list = single_table_info[1], single_table_info[3]
            if month == self.data.months[0]:
                self.current_query_table = self.creation_table_query_operations((sign, casual_condition_list),
                                                                                single_evaluation_table_3m)
            else:
                self.current_query_table = self.creation_table_query_operations((sign, casual_condition_list),
                                                                                single_evaluation_table_6m)
            self.current_total_amount = float("{:.2f}".format((self.current_query_table.AMOUNT.astype(np.float64)).sum()))
        self.op_features[j].append(self.operation_feature(self.current_total_amount, self.current_query_table, check_build_new_table_and_type, date, month, ndg))

    @staticmethod
    def multiple_causal_analytical(table, operations):
        if operations == 1: return operations

        conditions = (table.CAUSAL == operations[0])
        for i in range(1, len(operations)):
            conditions = conditions | (table.CAUSAL == operations[i])

        return conditions

    def reported_evaluation(self, ndg, date, months=24):
        return self.creation_table_query_evaluations((date, months, ndg)).shape[0]

    def operation_feature(self, total_amount, query_table, type_operation, data_anomaly=None, months=None, ndg=None):
        # type >> 0 = TOT, 1 = MEDIA, 2 = COUNT, 3 = RIPETITIVITA' PER MOV, 4 = RIPETITIVITA' PER NUM, 5 = MEDIA VERSAMENTO FILILARE, 6 = RIPETITIVITA' MOV PER FILIARE,
        # 7 = NUMERO DI FILIARI SU CUI SI HA OPERATO , 8 = RIPETITIVITA' NUM PER FILILARE, 10 = EVALUATION REPORTED
        if type_operation == 0: return total_amount
        elif type_operation == 1:
            if query_table.shape[0] == 0: return 0.0
            return total_amount / query_table.shape[0]
        elif type_operation == 2:
            return query_table.shape[0]
        elif type_operation == 3:
            return self.find_repetitiveness_movimenti(query_table)
        elif type_operation == 4:
            list_amounts, mean = self.mean_list_operation_per_value(query_table, data_anomaly, months)
            return self.find_repetitiveness_num(list_amounts, mean, self.data.variance_threshold_1,
                                                self.data.variance_threshold_2)
        elif type_operation == 10:
            return self.reported_evaluation(ndg, data_anomaly)

        if query_table.shape[0] == 0: return 0

        table_grouped_filiale = self.create_table_filial(query_table)
        if type_operation == 6:
            return table_grouped_filiale.AMOUNT.sum() / table_grouped_filiale.shape[0]
        elif type_operation == 7:
            return table_grouped_filiale.shape[0]
        elif type_operation == 8:
            return self.find_repetitiveness_movimenti(table_grouped_filiale)
        elif type_operation == 9:
            list_count = table_grouped_filiale.COUNT.values.tolist()
            return self.find_repetitiveness_num(list_count, sum(list_count) / len(list_count),
                                                self.data.variance_threshold_filiale, float('inf'))

    def accounts_feature(self, query_table, type_account_feature):
        # type >> 0 = NUM ACCOUNT EXPIRED, 1 = NUM ACCOUNT, 2 = ACCOUNT COINTESTATI
        if type_account_feature == 1:
            return query_table.EXPIRE_DATE.count()
        elif type_account_feature == 0:
            return query_table.START_DATE.count()
        elif type_account_feature == 2:
            num_account_cointestati = 0
            for code_account in query_table.CODE_ACCOUNT.values.tolist():
                if self.data.accounts[self.data.accounts.CODE_ACCOUNT == code_account].shape[0] > 1:
                    num_account_cointestati += 1
            return num_account_cointestati

    @staticmethod
    def create_table_filial(table):
        table_grouped_filiale = pd.DataFrame()
        table_grouped_filiale.insert(0, "FILIAL", table.FILIAL, True)
        table_grouped_filiale.insert(0, "AMOUNT",
                                     (table.AMOUNT.astype(np.float64).map('{:.2f}'.format)).astype(np.float64), True)
        table_grouped_filiale.insert(0, "COUNT", 1)
        return table_grouped_filiale.groupby(by=["FILIAL"]).sum()
