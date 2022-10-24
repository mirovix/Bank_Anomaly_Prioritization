#!/usr/bin/env python3

"""
@Author: Miro
@Date: 20/10/2022
@Version: 1.0
@Objective: definizione delle features legate all'anagrafica
@TODO:
"""

import numpy as np
from datetime import datetime


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
            if i >= self.data.max_elements: break

            if eval_country in self.data.country_risk['altissimo']:
                eval_country_list.append('ALTISSIMO')
            elif eval_country in self.data.country_risk['alto']:
                eval_country_list.append('ALTO')
            elif eval_country is np.nan:
                eval_country_list.append('NONE_CATEGORY')
            else:
                eval_country_list.append('MEDIO_BASSO')

        return eval_country_list

    def group(self, col, keys):
        group_values = []
        for i, evaluation in enumerate(self.evaluation_subjects_data[col.upper()].values.tolist()):
            flag_found = False
            if i >= self.data.max_elements: break

            for key in keys.keys():
                if evaluation in keys[key]:
                    group_values.append(key.upper())
                    flag_found = True
                    break

            if flag_found is True: continue

            if evaluation is np.nan:
                group_values.append(col.upper()+"_NONE")
            else:
                group_values.append(col.upper()+'_OTHERS')
        return group_values
