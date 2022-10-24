#!/usr/bin/env python3

"""
@Author: Miro
@Date: 20/10/2022
@Version: 1.0
@Objective: data per la creazione delle features
@TODO:
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from configs import build_features_config as bfc


class DataFeatures:
    def __init__(self, csv_data, max_elements, production,
                 subject_db, account_db, operations_db,
                 operations_day_db, target_db):

        self.production = production
        subjects = csv_data.load_subjects(subject_db)
        self.operations = csv_data.load_operations(operations_db)
        self.eval_subjects_comp, self.eval_subjects_day = csv_data.load_evaluation(target_db)
        if self.eval_subjects_comp.shape[0] > 0:
            self.eval_subjects_comp.drop_duplicates(inplace=True)
            self.eval_subjects_comp = pd.merge(self.eval_subjects_comp, subjects, on="NDG", how="left")

        if self.eval_subjects_day.shape[0] > 0:
            self.eval_subjects_day = pd.merge(self.eval_subjects_day, subjects, on="NDG", how="left")
            operations_day = csv_data.load_operations_day(operations_day_db)
            self.eval_subjects_day.IMPORTO = self.eval_subjects_day.IMPORTO.astype(np.float64)
            self.eval_subjects_day = pd.merge(self.eval_subjects_day, operations_day,
                                              how="left", left_on=['DATA_OPERATION', 'NDG', 'IMPORTO'],
                                              right_on=['DATE_OPERATION', 'NDG', 'AMOUNT'])

        self.accounts = csv_data.load_accounts(account_db)
        self.list_values = csv_data.load_list_values()
        self.list_causal_analytical = self.load_list_causal_analytical(csv_data.load_causal_analytical(),
                                                                       bfc.causali_version)
        self.causali_version = bfc.causali_version

        self.x = pd.DataFrame()
        self.y = pd.DataFrame()
        self.x_evaluated = pd.DataFrame()

        self.path_x = bfc.path_x
        self.path_y = bfc.path_y
        self.path_x_evaluated = bfc.path_x_evaluated

        self.prefix_op_names = bfc.prefix
        self.sign = ['D', 'A']
        self.months = bfc.months

        self.min_age = bfc.min_age
        self.max_age = bfc.max_age
        self.step_age = bfc.step_age

        self.range_repetitiveness = bfc.range_repetitiveness
        self.variance_threshold_1 = bfc.variance_threshold_1
        self.variance_threshold_2 = bfc.variance_threshold_2
        self.variance_threshold_filiale = bfc.variance_threshold_filiale

        self.max_elements = 0
        self.max_elements_comportamenti = self.set_max_elements(max_elements, self.eval_subjects_comp)
        self.max_elements_day = self.set_max_elements(max_elements, self.eval_subjects_day)

        self.country_risk = self.list_country_risk()
        self.prv = self.list_prov_residence()
        self.sae = self.list_sae()
        self.ateco = self.list_ateco()

    @staticmethod
    def set_max_elements(max_elements, eval_subject):
        if max_elements is None or eval_subject.shape[0] < max_elements + 1:
            return eval_subject.shape[0]
        return max_elements

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
