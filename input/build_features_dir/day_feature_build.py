#!/usr/bin/env python3

"""
@Author: Miro
@Date: 20/10/2022
@Version: 1.0
@Objective: costruzione delle features legate a discovery day
@TODO:
"""

from build_features_dir.evaluation_anagrafica_feature_build import EvaluationAnagraficaFeatureBuild
from build_features_dir.operations_accounts_feature_build import OperationsAccountsFeatureBuild


class DayFeatureBuild:
    def __init__(self, data):
        self.data = data
        self.anagrafica_features = EvaluationAnagraficaFeatureBuild(self.data, self.data.eval_subjects_day)
        self.operations_features = OperationsAccountsFeatureBuild(self.data, self.data.eval_subjects_day)

    def build(self):
        self.anagrafica_features.build()
        self.operations_features.build()
        self.day_features()

    def day_features(self):
        self.anagrafica_features.insert_registry_evaluation("CAUSALE", self.causal_categorization())

        self.anagrafica_features.insert_registry_evaluation("COUNTRY_OPERATION",
                                                            self.anagrafica_features.feature_definition.rischio_paese_residenza(
                                                                self.data.eval_subjects_day.COUNTRY))

        self.anagrafica_features.insert_registry_evaluation("SIGN", self.data.eval_subjects_day.SIGN)
        self.anagrafica_features.insert_registry_evaluation("AMOUNT_OPERATION",
                                                            self.data.eval_subjects_day.AMOUNT.values.tolist())
        self.anagrafica_features.insert_registry_evaluation("AMOUNT_OPERATION_CASH",
                                                            self.data.eval_subjects_day.AMOUNT_CASH.values.tolist())
        self.anagrafica_features.insert_registry_evaluation("COUNTRY_CONTROPARTE_OPERATION",
                                                            self.anagrafica_features.feature_definition.rischio_paese_residenza(
                                                                self.data.eval_subjects_day.COUNTERPART_SUBJECT_COUNTRY))
        self.anagrafica_features.insert_registry_evaluation("OPERATION_COUNTRY_EXECUTOR",
                                                            self.anagrafica_features.feature_definition.rischio_paese_residenza(
                                                                self.data.eval_subjects_day.RESIDENCE_COUNTRY_E))
        self.anagrafica_features.insert_registry_evaluation("RISK_PROFILE_EXECUTOR",
                                                            self.data.eval_subjects_day.RISK_PROFILE_E.values.tolist())

    def causal_categorization(self):
        list_causal = []
        for i, causal in enumerate(self.data.eval_subjects_day.CAUSAL.values.tolist()):
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
