#!/usr/bin/env python3

"""
@Author: Miro
@Date: 20/10/2022
@Version: 1.0
@Objective: definizione delle features legate alle operazioni e agli account
@TODO:
"""


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
