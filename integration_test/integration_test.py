#!/usr/bin/env python3

"""
@Author: Miro
@Date: 03/11/2022
@Version: 1.0
@Objective: test di integrazione per verificare il corretto funzionamento delle predizioni
@TODO:
"""

from extend_modules import extend_modules
extend_modules()

from input.build_features_dir.build_features import BuildFeatures
from pre_processing_features.categorization import Categorization
from productions.generation_fake_xml import generation_new_data
from input.load_data import LoadData
from productions.predict_anomaly import test_prediction
from productions.production import connections, execute as ex_production
from configs import production_config as pc
import unittest

pc.max_elements_test = 40
pc.max_elements = None


class TestPredictionMethods(unittest.TestCase):
    def test_integration(self):
        connections()
        target = generation_new_data(pc.engine_rx_input, pc.max_elements_test)
        db_prediction, _ = ex_production()

        features = BuildFeatures(LoadData(), target_db=target, production=True)
        x_dataset_cat = Categorization(features.get_dataset()).run_production()
        local_prediction = test_prediction(x=x_dataset_cat, num_elements=None)

        if local_prediction.shape[0] + db_prediction.shape[0] < 1: return True

        local_prediction.sort_index(inplace=True)
        db_prediction.sort_index(inplace=True)

        print("\nlocal prediction ", local_prediction.shape[0])
        print("db prediction ", db_prediction.shape[0])

        comparison = local_prediction.compare(db_prediction)
        print(comparison)

        self.assertEqual(local_prediction.shape[0], db_prediction.shape[0])
        self.assertEqual(comparison.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
