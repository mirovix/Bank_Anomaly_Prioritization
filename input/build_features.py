#!/usr/bin/env python3

"""
@Author: Miro
@Date: 06/05/2022
@Version: 1.2
@Objective: creazione delle features
@TODO: print information about day-comportamenti
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta


class BuildFeatures:
    def __init__(self, csv_data, max_elements=None, prefix=None, months=None, range_repetitiveness=0.05,
                 variance_threshold_1=0.80, variance_threshold_2=0.75, variance_threshold_filiale=35,
                 min_age=11, max_age=110, step_age=10, causali_version=14,
                 path_x="../data/dataset_x.csv", path_y="../data/dataset_y.csv",
                 path_x_evaluated="../data/dataset_x_evaluated.csv", production=False):
        start = datetime.now()
        if months is None:
            months = [3, 6]
        if prefix is None:
            prefix = ["tot", "media", "num", "ripetitività_mov", "ripetitività_num"]

        self.data = DataFeatures(csv_data, max_elements, prefix, months, range_repetitiveness,
                                 variance_threshold_1, variance_threshold_2, variance_threshold_filiale,
                                 min_age, max_age, step_age, causali_version,
                                 path_x, path_y, path_x_evaluated, production)

        print(">> csv files loaded in ", datetime.now() - start)
        self.print_info()

    def print_info(self):
        print("\n>> GENERAL INFO")
        print(">> dataset size comportamenti", self.data.max_elements_comportamenti)
        print(">> dataset size day", self.data.max_elements_day)
        print(">> total evaluations loaded comportamenti", self.data.evaluations_subjects_comportamenti.shape[0])
        print(">> total evaluations loaded day", self.data.evaluations_subjects_day.shape[0])
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
        if max_elements is None:
            self.data.max_elements = self.data.max_elements_day
        else:
            self.data.max_elements = max_elements
        start = datetime.now()
        print("\n>> DISCOVERY DAY AND ACCOUNTS FEATURES")
        DayFeatureBuild(self.data).build()
        print("   >> discovery day and accounts features done in ", datetime.now() - start)

    def get_dataset_discovery_comportamenti(self, max_elements=None):
        if max_elements is None:
            self.data.max_elements = self.data.max_elements_comportamenti
        else:
            self.data.max_elements = max_elements

        start = datetime.now()
        print("\n>> OPERATIONS AND ACCOUNTS FEATURES")
        OperationsAccountsFeatureBuild(self.data, self.data.evaluations_subjects_comportamenti).build()
        print("   >> operations and accounts features done in ", datetime.now() - start)

        start = datetime.now()
        print("\n>> ANAGRAFICA AND EVALUATION FEATURES")
        EvaluationAnagraficaFeatureBuild(self.data, self.data.evaluations_subjects_comportamenti).build()
        print("   >> anagrafica and evaluation features done in ", datetime.now() - start)

    def get_dataset_day_comportamenti(self, elements_day, elements_comp):
        self.get_dataset_discovery_comportamenti(max_elements=elements_comp)
        comportamenti_x, comportamenti_y = self.data.x, self.data.y
        self.data.x, self.data.y = pd.DataFrame(), pd.DataFrame()

        self.get_dataset_discovery_day(max_elements=elements_day)
        if self.data.production is False:
            self.data.y.replace(to_replace=0, value=2, inplace=True)
            self.data.y.replace(to_replace=1, value=3, inplace=True)
        day_col_x = self.data.x.columns

        self.data.x = pd.concat([comportamenti_x, self.data.x], ignore_index=True, sort=False)
        self.data.y = pd.concat([comportamenti_y, self.data.y], ignore_index=True, sort=False)

        fill_with_string = ['AMOUNT_OPERATION', 'AMOUNT_OPERATION_CASH', 'RISK_PROFILE_EXECUTOR']
        self.fill_identificativi_vari(fill_with_string, comportamenti_x, day_col_x)
        for col in self.data.x.columns:
            if (col in comportamenti_x.columns and col not in day_col_x) or \
                    (col in day_col_x and col not in comportamenti_x.columns):
                if col in fill_with_string:
                    self.data.x[col].fillna(-1, inplace=True, limit=self.data.max_elements_comportamenti)
                else:
                    self.data.x[col].fillna("OTHER_DB", inplace=True, limit=self.data.max_elements_comportamenti)
        return

    def fill_identificativi_vari(self, list_int, comportamenti_x, day_col_x):
        for i, operation in enumerate(self.data.evaluations_subjects_day.CODE_OPERATION.values.tolist()):
            if i >= self.data.max_elements:
                break
            if operation == 'Identificativi vari':
                for col in self.data.x.columns:
                    if (col in comportamenti_x.columns and col not in day_col_x) or \
                            (col in day_col_x and col not in comportamenti_x.columns):
                        if col in list_int:
                            self.data.x.at[i, col] = -2
                        else:
                            self.data.x.at[i, col] = "IDENT_VARI"

    def get_dataset(self, discovery_day=False, discovery_comportamenti=False):
        elements_day, elements_comp = None, None
        if discovery_day is True and discovery_comportamenti is False:
            elements_comp, elements_day = 0, None
        elif discovery_day is False and discovery_comportamenti is True:
            elements_day, elements_comp = 0, None
        self.get_dataset_day_comportamenti(elements_day, elements_comp)
        self.data.x.set_index('ID', inplace=True)
        if self.data.production is False:
            return self.data.x, self.data.y
        else:
            return self.data.x

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


class DayFeatureBuild:
    def __init__(self, data):
        self.data = data
        self.anagrafica_features = EvaluationAnagraficaFeatureBuild(self.data, self.data.evaluations_subjects_day)
        self.operations_features = OperationsAccountsFeatureBuild(self.data, self.data.evaluations_subjects_day)

    def build(self):
        self.anagrafica_features.build()
        self.operations_features.build()
        self.day_features()

    def day_features(self):
        self.anagrafica_features.insert_registry_evaluation("CAUSALE", self.causal_categorization())

        self.anagrafica_features.insert_registry_evaluation("COUNTRY_OPERATION",
                                                            self.anagrafica_features.feature_definition.rischio_paese_residenza(
                                                                self.data.evaluations_subjects_day.COUNTRY))

        self.anagrafica_features.insert_registry_evaluation("SIGN", self.data.evaluations_subjects_day.SIGN)
        self.anagrafica_features.insert_registry_evaluation("AMOUNT_OPERATION",
                                                            self.data.evaluations_subjects_day.AMOUNT.values.tolist())
        self.anagrafica_features.insert_registry_evaluation("AMOUNT_OPERATION_CASH",
                                                            self.data.evaluations_subjects_day.AMOUNT_CASH.values.tolist())
        self.anagrafica_features.insert_registry_evaluation("COUNTRY_CONTROPARTE_OPERATION",
                                                            self.anagrafica_features.feature_definition.rischio_paese_residenza(
                                                                self.data.evaluations_subjects_day.COUNTERPART_SUBJECT_COUNTRY))
        self.anagrafica_features.insert_registry_evaluation("OPERATION_COUNTRY_EXECUTOR",
                                                            self.anagrafica_features.feature_definition.rischio_paese_residenza(
                                                                self.data.evaluations_subjects_day.RESIDENCE_COUNTRY_E))
        self.anagrafica_features.insert_registry_evaluation("RISK_PROFILE_EXECUTOR",
                                                            self.data.evaluations_subjects_day.RISK_PROFILE_E.values.tolist())

    def causal_categorization(self):
        list_causal = []
        for i, causal in enumerate(self.data.evaluations_subjects_day.CAUSAL.values.tolist()):
            if i >= self.data.max_elements:
                break
            flag_found = False
            for key in self.data.list_causal_analytical:
                if causal in self.data.list_causal_analytical[key]:
                    list_causal.append(key)
                    flag_found = True
                    break
            if flag_found is False:
                list_causal.append('others')
        return list_causal


class EvaluationAnagraficaFeatureBuild:
    def __init__(self, data, evaluation_subjects_data):
        self.data = data
        self.evaluation_subjects_data = evaluation_subjects_data
        self.feature_definition = EvaluationAnagraficaFeatureDefinition(self.data, self.evaluation_subjects_data)

    def build(self):
        self.evaluation_feature()
        self.anagrafica_feature()

    def anagrafica_feature(self):
        self.insert_registry_evaluation("AGE_CLUSTER", self.feature_definition.age_feature())

        self.insert_registry_evaluation("REPORTED_6_MONTHS", self.check_last_anomalie())

        self.insert_registry_evaluation("LEGAL_SPECIE", self.evaluation_subjects_data.LEGAL_SPECIE)

        # self.insert_anagrafica("RESIDENCE_PROVINCE", self.evaluations_subjects.RESIDENCE_PROVINCE)
        self.insert_registry_evaluation("RESIDENCE_PROVINCE", self.feature_definition.group_province_residenza())

        # self.insert_anagrafica("RESIDENCE_COUNTRY", self.evaluations_subjects.RESIDENCE_COUNTRY)
        self.insert_registry_evaluation("RISK_RESIDENCE_COUNTRY", self.feature_definition.rischio_paese_residenza(
            self.evaluation_subjects_data.RESIDENCE_COUNTRY))

        # self.insert_anagrafica("SAE", self.evaluations_subjects.SAE)
        self.insert_registry_evaluation("SAE", self.feature_definition.group_sae())

        # self.insert_anagrafica("ATECO", self.evaluations_subjects.ATECO)
        self.insert_registry_evaluation("ATECO", self.feature_definition.group_ateco())

        self.insert_registry_evaluation("RISK_PROFILE", self.evaluation_subjects_data.RISK_PROFILE)

        self.insert_registry_evaluation("REPORTED", self.evaluation_subjects_data.REPORTED)

        # self.insert_registry_evaluation("PREJUDICIAL", self.data.evaluations_subjects.PREJUDICIAL.astype(np.int8))

        self.insert_registry_evaluation("GROSS_INCOME", self.evaluation_subjects_data.GROSS_INCOME)

        self.insert_registry_evaluation("NCHECKREQUIRED", self.evaluation_subjects_data.NCHECKREQUIRED)
        self.insert_registry_evaluation("NCHECKDEBITED", self.evaluation_subjects_data.NCHECKDEBITED)
        self.insert_registry_evaluation("NCHECKAVAILABLE", self.evaluation_subjects_data.NCHECKAVAILABLE)

        self.insert_registry_evaluation("DATA", self.evaluation_subjects_data.DATA)

    def evaluation_feature(self):
        self.insert_registry_evaluation('IMPORTO', self.evaluation_subjects_data.IMPORTO)
        self.insert_registry_evaluation('CODICE_ANOMALIA', self.evaluation_subjects_data.CODICE_ANOMALIA)
        self.insert_registry_evaluation('ID', self.evaluation_subjects_data.ID)
        if self.data.production is False:
            self.data.y.insert(0, "EVALUATION", self.evaluation_subjects_data.STATO.astype(np.int8).values.tolist()[
                                                :self.data.max_elements], True)

    def creation_table_query_evaluations(self, query_prop):
        data_anomaly, num_months, ndg = query_prop
        start_range, end_range = range_date(data_anomaly, num_months)
        query_conditions = (self.evaluation_subjects_data.DATA < str(start_range)) & (
                self.evaluation_subjects_data.DATA > str(end_range)) & (self.evaluation_subjects_data.NDG == ndg)
        return self.evaluation_subjects_data[query_conditions]

    def check_last_anomalie(self):
        values = []
        for i, row in self.evaluation_subjects_data.iterrows():
            if i >= self.data.max_elements:
                break
            data_anomaly, num_months, ndg = row['DATA'], 6, row['NDG']
            query_table = self.creation_table_query_evaluations((data_anomaly, num_months, ndg))
            values.append(query_table.shape[0])
        return values

    def insert_registry_evaluation(self, name, list_values):
        temp_df = pd.DataFrame(list_values[:self.data.max_elements], columns=[name])
        self.data.x = pd.concat([self.data.x, temp_df], axis=1)


class EvaluationAnagraficaFeatureDefinition:
    def __init__(self, data, evaluation_subjects_data):
        self.data = data
        self.evaluation_subjects_data = evaluation_subjects_data

    @staticmethod
    def age_computing(birth):
        if not isinstance(birth, str):
            return birth
        born = datetime.fromisoformat(birth)
        today = datetime.now()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

    def age_feature(self):
        all_age_features = []
        for count, birth in enumerate(self.evaluation_subjects_data.BIRTH_DAY.values.tolist()):
            if count >= self.data.max_elements:
                break
            age = self.age_computing(birth)
            value = 0
            flag_age_computed = 0
            for i in range(self.data.min_age, self.data.max_age, self.data.step_age):
                if age < i:
                    all_age_features.append(str(value) + "_CATEGORY")
                    flag_age_computed = 1
                    break
                value += 1
            if flag_age_computed == 0:
                all_age_features.append("NONE_CATEGORY")

        return all_age_features

    def rischio_paese_residenza(self, country_data):
        eval_country_list = []
        for i, eval_country in enumerate(country_data.values.tolist()):
            if i >= self.data.max_elements:
                break
            if eval_country in self.data.country_risk['altissimo']:
                eval_country_list.append('ALTISSIMO')
            elif eval_country in self.data.country_risk['alto']:
                eval_country_list.append('ALTO')
            elif eval_country is np.nan:
                eval_country_list.append('NONE_CATEGORY')
            else:
                eval_country_list.append('MEDIO_BASSO')

        return eval_country_list

    def group_province_residenza(self):
        grop_province = []
        for i, evaluation in enumerate(self.evaluation_subjects_data.RESIDENCE_PROVINCE.values.tolist()):
            if i >= self.data.max_elements:
                break
            if evaluation in self.data.prv['prv_0']:
                grop_province.append('PRV_0')
            elif evaluation in self.data.prv['prv_1']:
                grop_province.append('PRV_1')
            elif evaluation in self.data.prv['prv_2']:
                grop_province.append('PRV_2')
            elif evaluation in self.data.prv['prv_3']:
                grop_province.append('PRV_3')
            elif evaluation in self.data.prv['prv_4']:
                grop_province.append('PRV_4')
            elif evaluation is np.nan:
                grop_province.append('PRV_NONE')
            else:
                grop_province.append('OTHERS')
        return grop_province

    def group_ateco(self):
        grop_ateco = []
        for i, evaluation in enumerate(self.evaluation_subjects_data.ATECO.values.tolist()):
            if i >= self.data.max_elements:
                break
            if evaluation in self.data.ateco['ateco_0']:
                grop_ateco.append('ATECO_0')
            elif evaluation in self.data.ateco['ateco_1']:
                grop_ateco.append('ATECO_1')
            elif evaluation in self.data.ateco['ateco_2']:
                grop_ateco.append('ATECO_2')
            elif evaluation in self.data.ateco['ateco_3']:
                grop_ateco.append('ATECO_3')
            elif evaluation is np.nan:
                grop_ateco.append('ATECO_NONE')
            else:
                grop_ateco.append('OTHERS')
        return grop_ateco

    def group_sae(self):
        group_sae = []
        for i, evaluation in enumerate(self.evaluation_subjects_data.SAE.values.tolist()):
            if i >= self.data.max_elements:
                break
            if evaluation in self.data.sae['sae_0']:
                group_sae.append('SAE_0')
            elif evaluation in self.data.sae['sae_1']:
                group_sae.append('SAE_1')
            elif evaluation in self.data.sae['sae_2']:
                group_sae.append('SAE_2')
            elif evaluation in self.data.sae['sae_3']:
                group_sae.append('SAE_3')
            elif evaluation in self.data.sae['sae_4']:
                group_sae.append('SAE_4')
            elif evaluation is np.nan:
                group_sae.append("SAE_NONE")
            else:
                group_sae.append('OTHERS')
        return group_sae


class OperationsAccountsFeatureBuild:
    def __init__(self, data, evaluation_subjects_data):
        self.data = data
        self.evaluation_subjects_data = evaluation_subjects_data
        self.operations_accounts_feature_data = OperationsAccountsFeatureDefinition(
            self.data).operations_accounts_feature()
        self.op_features = None
        self.current_query_table = None
        self.current_total_amount = 0

    def build(self):
        self.insert_table_operation_account(self.operations_accounts_feature_data)

    def creation_table_query_evaluations(self, query_prop):
        data_anomaly, num_months, ndg = query_prop
        start_range, end_range = range_date(data_anomaly, num_months)
        query_conditions = (self.evaluation_subjects_data.DATA < str(start_range)) & (
                self.evaluation_subjects_data.DATA > str(end_range)) & (
                                   self.evaluation_subjects_data.NDG == ndg)
        return self.evaluation_subjects_data[query_conditions]

    def creation_table_query_operations(self, query_prop, table):
        sign, casual_condition_list = query_prop
        casual_condition = self.multiple_causal_analytical(table, casual_condition_list)
        query_conditions = (table.SIGN == sign) & casual_condition
        return table[query_conditions]

    @staticmethod
    def creation_initial_query_table_ndg(table, date, max_months, ndg_condition, type_operations_account=1):
        start_range, end_range = range_date(date, max_months)
        if type_operations_account == 1:
            date = table.DATE_OPERATION
        else:
            date = table.START_DATE

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
            if max_rip < current_rip:
                max_rip = current_rip + 1
        return max_rip

    @staticmethod
    def mean_list_operation_per_value(query_table, data_anomaly, months):
        rip_elements = []
        for m in range(months):
            start_range, end_range = range_date(data_anomaly, (m + 1))
            start_range -= relativedelta(months=m)
            query_conditions = (query_table.DATE_OPERATION < str(start_range)) & (
                    query_table.DATE_OPERATION > str(end_range))
            rip_elements.append(query_table[query_conditions].shape[0])

        media_soglia1 = sum(rip_elements) / months
        return rip_elements, media_soglia1

    @staticmethod
    def find_repetitiveness_num(rip_elements, media_soglia1, th1, th2):
        rip_elements_processed = []
        max_rip, media_soglia2 = 0, 0
        if len(rip_elements) == 0:
            return 0

        for e in rip_elements:
            soglia = e * th1
            if abs(media_soglia1 - e) ** 2 < soglia and e > 0:
                rip_elements_processed.append(e)

        if len(rip_elements_processed) == 0:
            return 0

        media_soglia2 = sum(rip_elements_processed) / len(rip_elements_processed)
        soglia = media_soglia2 * th2
        for e in rip_elements_processed:
            if abs(media_soglia2 - e) ** 2 < soglia:
                max_rip += 1

        if max_rip < 2:
            max_rip = 0

        return max_rip

    def insert_table_operation_account(self, all_tables_information):
        self.op_features = [[] for _ in range(len(all_tables_information))]
        for i, evaluation in enumerate(self.evaluation_subjects_data.values.tolist()):

            if i >= self.data.max_elements:
                break

            to_write = "   >> evaluations completed " + str(i + 1) + "/" + str(self.data.max_elements) + "\r"
            sys.stdout.write(to_write)

            single_evaluation_table_6m = self.creation_initial_query_table_ndg(
                table=self.data.operations, date=evaluation[4], max_months=self.data.months[1],
                ndg_condition=(self.data.operations.NDG == evaluation[6]), type_operations_account=1)
            single_evaluation_table_3m = self.creation_initial_query_table_ndg(
                table=single_evaluation_table_6m, date=evaluation[4], max_months=self.data.months[0], ndg_condition=1,
                type_operations_account=1)
            single_evaluation_table_6m_account = self.creation_initial_query_table_ndg(
                table=self.data.accounts, date=evaluation[4], max_months=self.data.months[1],
                ndg_condition=(self.data.accounts.NDG == evaluation[6]), type_operations_account=0)
            single_evaluation_table_3m_account = self.creation_initial_query_table_ndg(
                table=single_evaluation_table_6m_account, date=evaluation[4], max_months=self.data.months[0],
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
        date = evaluation[4]
        month = single_table_info[2]
        ndg = evaluation[6]
        if check_build_new_table_and_type == 0 or check_build_new_table_and_type == 6:
            sign, casual_condition_list = single_table_info[1], single_table_info[3]
            if month == self.data.months[0]:
                self.current_query_table = self.creation_table_query_operations((sign, casual_condition_list),
                                                                                single_evaluation_table_3m)
            else:
                self.current_query_table = self.creation_table_query_operations((sign, casual_condition_list),
                                                                                single_evaluation_table_6m)
            self.current_total_amount = float(
                "{:.2f}".format((self.current_query_table.AMOUNT.astype(np.float64)).sum()))
        self.op_features[j].append(
            self.operation_feature(self.current_total_amount, self.current_query_table, check_build_new_table_and_type,
                                   date, month, ndg))

    @staticmethod
    def multiple_causal_analytical(table, operations):
        if operations == 1:
            return operations
        conditions = (table.CAUSAL == operations[0])
        for i in range(1, len(operations)):
            conditions = conditions | (table.CAUSAL == operations[i])
        return conditions
        # return [conditions |= (table.CAUSAL == operations[i]) for i in range(1, len(operations))]

    def reported_evaluation(self, ndg, date, months=24):
        return self.creation_table_query_evaluations((date, months, ndg)).shape[0]

    def operation_feature(self, total_amount, query_table, type_operation, data_anomaly=None, months=None, ndg=None):
        # type >> 0 = TOT, 1 = MEDIA, 2 = COUNT, 3 = RIPETITIVITA' PER MOV, 4 = RIPETITIVITA' PER NUM, 5 = MEDIA VERSAMENTO FILILARE, 6 = RIPETITIVITA' MOV PER FILIARE,
        # 7 = NUMERO DI FILIARI SU CUI SI HA OPERATO , 8 = RIPETITIVITA' NUM PER FILILARE, 10 = EVALUATION REPORTED
        if type_operation == 0:
            return total_amount
        elif type_operation == 1:
            if query_table.shape[0] == 0:
                return 0.0
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

        if query_table.shape[0] == 0:
            return 0

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


class OperationsAccountsFeatureDefinition:
    def __init__(self, data):
        self.data = data
        self.id_operations = 0
        self.id_accounts = 1
        self.all_features_operations_accounts = []

    @staticmethod
    def multiple_causal_names(operations):
        name_op = operations[0]
        for i in range(1, len(operations)):
            name_op += '-' + operations[i]
        return name_op

    def operations_all_causal(self):
        type_operation = 0
        ndg_condition_always_true = 1
        for s in self.data.sign:
            for m in self.data.months:
                name = "all_movimentazioni" + "_sign-" + s + "_months-" + str(m)
                self.all_features_operations_accounts.append(
                    [name, s, m, ndg_condition_always_true, type_operation, self.id_operations])

    def operations_feature_versamento_filiale(self, operations):
        name_op = self.multiple_causal_names(operations)
        name_feature = "_filiale"
        for s in self.data.sign:
            for m in self.data.months:
                base_name = "_sign-" + s + "_operation-" + name_op + "_months-" + str(m)
                for i in range(1, len(self.data.prefix_op_names)):
                    self.all_features_operations_accounts.append(
                        [self.data.prefix_op_names[i] + name_feature + base_name, s, m, operations, i + 5,
                         self.id_operations])

    def expired_accounts(self):
        type_operation = 1
        default_value = 0
        for m in self.data.months:
            name = "expired_accounts_" + str(m)
            self.all_features_operations_accounts.append(
                [name, m, type_operation, default_value, default_value, self.id_accounts])

    def num_accounts_opened(self):
        default_value = 0
        type_operation = 0
        for m in self.data.months:
            name = "num_accounts_" + str(m)
            self.all_features_operations_accounts.append(
                [name, m, type_operation, default_value, default_value, self.id_accounts])

    def num_accounts_cointestati(self):
        months = 6
        default_value = 0
        type_operation = 2
        name = "num_accounts_cointestati_" + str(months)
        self.all_features_operations_accounts.append(
            [name, months, type_operation, default_value, default_value, self.id_accounts])

    def operations_features(self, key):
        operations = self.data.list_causal_analytical[key]
        name_op = self.multiple_causal_names(operations)
        name_feature = "_" + key
        for s in self.data.sign:
            for m in self.data.months:
                base_name = "_sign-" + s + "_operation-" + name_op + "_months-" + str(m)
                for i in range(len(self.data.prefix_op_names)):
                    self.all_features_operations_accounts.append(
                        [self.data.prefix_op_names[i] + name_feature + base_name, s, m, operations, i,
                         self.id_operations])

    def reported_evaluation_feature(self, name_feature):
        self.all_features_operations_accounts.append(
            [name_feature, None, None, None, 10, self.id_operations])

    def operations_accounts_feature(self):
        self.operations_all_causal()

        for key in self.data.list_causal_analytical:
            self.operations_features(key)

        self.operations_feature_versamento_filiale(self.data.list_causal_analytical['contante'])
        self.reported_evaluation_feature("reported_evaluation")

        self.num_accounts_opened()
        self.expired_accounts()
        self.num_accounts_cointestati()
        return self.all_features_operations_accounts


class DataFeatures:
    def __init__(self, csv_data, max_elements, prefix, months, range_repetitiveness,
                 variance_threshold_1, variance_threshold_2, variance_threshold_filiale,
                 min_age, max_age, step_age, causali_version,
                 path_x, path_y, path_x_evaluated, production):
        self.production = production
        eval_comportamenti, eval_day = csv_data.load_evaluation()
        self.evaluations_subjects_comportamenti = pd.merge(eval_comportamenti, csv_data.load_subjects(), on="NDG",
                                                           how="left")

        self.operations = csv_data.load_operations()

        self.evaluations_subjects_day = pd.merge(eval_day, csv_data.load_subjects(), on="NDG", how="left")
        operations_day = csv_data.load_operations_day()
        cols_to_use = operations_day.columns.difference(self.evaluations_subjects_day.columns).tolist()
        cols_to_use.append('CODE_OPERATION')
        self.evaluations_subjects_day = pd.merge(self.evaluations_subjects_day, operations_day[cols_to_use],
                                                 on="CODE_OPERATION", how="left")

        self.accounts = csv_data.load_accounts()
        self.list_values = csv_data.load_list_values()
        self.list_causal_analytical = self.load_list_causal_analytical(csv_data.load_causal_analytical(),
                                                                       causali_version)
        self.causali_version = causali_version

        self.x = pd.DataFrame()
        self.y = pd.DataFrame()
        self.x_evaluated = pd.DataFrame()

        self.path_x = path_x
        self.path_y = path_y
        self.path_x_evaluated = path_x_evaluated

        self.prefix_op_names = prefix
        self.sign = ['D', 'A']
        self.months = months

        self.min_age = min_age
        self.max_age = max_age
        self.step_age = step_age

        self.range_repetitiveness = range_repetitiveness
        self.variance_threshold_1 = variance_threshold_1
        self.variance_threshold_2 = variance_threshold_2
        self.variance_threshold_filiale = variance_threshold_filiale

        self.max_elements = 0
        self.max_elements_comportamenti = self.set_max_elements(max_elements, self.evaluations_subjects_comportamenti)
        self.max_elements_day = self.set_max_elements(max_elements, self.evaluations_subjects_day)

        self.country_risk = self.list_country_risk()
        self.prv = self.list_prov_residence()
        self.sae = self.list_sae()
        self.ateco = self.list_ateco()

    @staticmethod
    def set_max_elements(max_elements, eval_subject):
        if max_elements is not None:
            return max_elements
        return eval_subject.shape[0]

    @staticmethod
    def load_list_causal_analytical(causal_analytical_df, version):
        if version == 21:
            return {
                "contante": causal_analytical_df.COD_CONTANTE.dropna().values.tolist(),
                "assegni": causal_analytical_df.COD_ASSEGNI.dropna().values.tolist(),
                "atm": causal_analytical_df.COD_ATM.dropna().values.tolist(),
                "finanziamenti": causal_analytical_df.COD_FINANZIAMENTI.dropna().values.tolist(),
                "cert_deposito": causal_analytical_df.COD_CER_DEPOSITO.dropna().values.tolist(),
                "bonifici_domestici": causal_analytical_df.COD_BONIFICI_DOMESTICI.dropna().values.tolist(),
                "bonifici_esteri": causal_analytical_df.COD_BONIFICI_ESTERI.dropna().values.tolist(),
                "titoli": causal_analytical_df.COD_TITOLI.dropna().values.tolist(),
                "pos": causal_analytical_df.COD_POS.dropna().values.tolist(),
                "pag_inc_diversi": causal_analytical_df.COD_PAG_INC_DIVERSI.dropna().values.tolist(),
                "emolumenti": causal_analytical_df.COD_EMOLUMENTI.dropna().values.tolist(),
                "valuta_estera": causal_analytical_df.COD_VALUTA_ESTERA.dropna().values.tolist(),
                "effetti": causal_analytical_df.COD_EFFETTI.dropna().values.tolist(),
                "documenti": causal_analytical_df.COD_DOCUMENTI.dropna().values.tolist(),
                "dividendi": causal_analytical_df.COD_DIVIDENDI.dropna().values.tolist(),
                "import_export": causal_analytical_df.COD_IMPORT_EXPORT.dropna().values.tolist(),
                "metallo": causal_analytical_df.COD_METALLO.dropna().values.tolist(),
                "reversali": causal_analytical_df.COD_REVERSALI.dropna().values.tolist(),
                "rimesse": causal_analytical_df.COD_RIMESSE.dropna().values.tolist(),
                "invest": causal_analytical_df.COD_INVEST.dropna().values.tolist(),
                "riba": causal_analytical_df.COD_RIBA.dropna().values.tolist()
            }
        elif version == 14:
            return {
                "contante": causal_analytical_df.COD_CONTANTE.dropna().values.tolist(),

                "assegni": causal_analytical_df.COD_ASSEGNI.dropna().values.tolist(),
                "atm": causal_analytical_df.COD_ATM.dropna().values.tolist(),
                "finanziamenti": causal_analytical_df.COD_FINANZIAMENTI.dropna().values.tolist(),

                "bonifici_domestici": causal_analytical_df.COD_BONIFICI_DOMESTICI.dropna().values.tolist(),
                "bonifici_esteri": causal_analytical_df.COD_BONIFICI_ESTERI.dropna().values.tolist(),

                "pos": causal_analytical_df.COD_POS.dropna().values.tolist(),
                "pag_inc_diversi": causal_analytical_df.COD_PAG_INC_DIVERSI.dropna().values.tolist(),

                "dividendi": causal_analytical_df.COD_DIVIDENDI.dropna().values.tolist(),

                "metallo": causal_analytical_df.COD_METALLO.dropna().values.tolist(),
                "reversali": causal_analytical_df.COD_REVERSALI.dropna().values.tolist(),
                "rimesse": causal_analytical_df.COD_RIMESSE.dropna().values.tolist(),

                "eff_doc_riba": causal_analytical_df.COD_EFF_DOC_RIBA.dropna().values.tolist(),

                "tit_cer_inv": causal_analytical_df.COD_TIT_CER_INV.dropna().values.tolist()
            }

    def list_sae(self):
        return {
            "sae_0": self.list_values.SAE_0.dropna().values.tolist(),
            "sae_1": self.list_values.SAE_1.dropna().values.tolist(),
            "sae_2": self.list_values.SAE_2.dropna().values.tolist(),
            "sae_3": self.list_values.SAE_3.dropna().values.tolist(),
            "sae_4": self.list_values.SAE_4.dropna().values.tolist()
        }

    def list_ateco(self):
        return {
            "ateco_0": self.list_values.ATECO_0.dropna().values.tolist(),
            "ateco_1": self.list_values.ATECO_1.dropna().values.tolist(),
            "ateco_2": self.list_values.ATECO_2.dropna().values.tolist(),
            "ateco_3": self.list_values.ATECO_3.dropna().values.tolist(),
            "ateco_sae_none": self.list_values.ATECO_SAE_NONE.dropna().values.tolist()
        }

    def list_prov_residence(self):
        return {
            "prv_0": self.list_values.PRV_0.dropna().values.tolist(),
            "prv_1": self.list_values.PRV_1.dropna().values.tolist(),
            "prv_2": self.list_values.PRV_2.dropna().values.tolist(),
            "prv_3": self.list_values.PRV_3.dropna().values.tolist(),
            "prv_4": self.list_values.PRV_4.dropna().values.tolist()
        }

    def list_country_risk(self):
        return {
            "altissimo": self.list_values.RISCHIO_PAESE_ALTISSIMO.dropna().values.tolist(),
            "alto": self.list_values.RISCHIO_PAESE_ALTO.dropna().values.tolist(),
            "basso_medio": []
        }


def range_date(data_anomaly, num_months, months_to_remove=1):
    data_anomaly = datetime.fromisoformat(data_anomaly)
    start_range = data_anomaly - relativedelta(months=months_to_remove, days=data_anomaly.day - 1,
                                               hours=data_anomaly.hour, minutes=data_anomaly.minute,
                                               seconds=data_anomaly.second,
                                               microseconds=data_anomaly.microsecond)
    return start_range, (start_range - relativedelta(months=num_months))
