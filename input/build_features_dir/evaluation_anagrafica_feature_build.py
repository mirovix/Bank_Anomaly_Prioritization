#!/usr/bin/env python3

"""
@Author: Miro
@Date: 20/10/2022
@Version: 1.0
@Objective: costruzione delle features legate all'anagrafica
@TODO:
"""

import numpy as np
import pandas as pd
from data_features import range_date
from build_features_dir.evaluation_anagrafica_features_definition import EvaluationAnagraficaFeatureDefinition as eafd


class EvaluationAnagraficaFeatureBuild:
    def __init__(self, input_data, evaluation_subjects_data):
        self.data = input_data
        self.evaluation_subjects_data = evaluation_subjects_data
        self.feature_definition = eafd(self.data, self.evaluation_subjects_data)

    def build(self):
        self.evaluation_feature()
        self.anagrafica_feature()

    def anagrafica_feature(self):
        self.insert_registry_evaluation("AGE_CLUSTER", self.feature_definition.age_feature())

        self.insert_registry_evaluation("REPORTED_6_MONTHS", self.check_last_anomalie())

        self.insert_registry_evaluation("LEGAL_SPECIE", self.evaluation_subjects_data.LEGAL_SPECIE)

        self.insert_registry_evaluation("RESIDENCE_PROVINCE", self.feature_definition.group("residence_province", self.data.prv))

        self.insert_registry_evaluation("RISK_RESIDENCE_COUNTRY", self.feature_definition.rischio_paese_residenza(
            self.evaluation_subjects_data.RESIDENCE_COUNTRY))

        self.insert_registry_evaluation("SAE", self.feature_definition.group("sae", self.data.sae))

        self.insert_registry_evaluation("ATECO", self.feature_definition.group("ateco", self.data.ateco))

        self.insert_registry_evaluation("RISK_PROFILE", self.evaluation_subjects_data.RISK_PROFILE)

        self.insert_registry_evaluation("REPORTED", self.evaluation_subjects_data.REPORTED)

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
            if i >= self.data.max_elements: break
            data_anomaly, num_months, ndg = row['DATA'], 6, row['NDG']
            query_table = self.creation_table_query_evaluations((data_anomaly, num_months, ndg))
            values.append(query_table.shape[0])
        return values

    def insert_registry_evaluation(self, name, list_values):
        temp_df = pd.DataFrame(list_values[:self.data.max_elements], columns=[name])
        self.data.x = pd.concat([self.data.x, temp_df], axis=1)
