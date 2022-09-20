#!/usr/bin/env python3

"""
@Author: Miro
@Date: 12/09/2022
@Version: 1.12
@Objective: input test del web service
@TODO:
"""

import pickle
import base64
from datetime import datetime
import requests
from build_features import BuildFeatures
from categorization import Categorization
from load_csv import LoadData


def run_production(max_elements=5):
    start = datetime.now()
    csv_files = LoadData(max_months_considered=26)
    features = BuildFeatures(csv_files, production=True, max_elements=max_elements)
    x_dataset = features.get_dataset(discovery_day=True, discovery_comportamenti=True)
    x_dataset_cat = Categorization(x_dataset).run_production()

    x_dataset_not_processed = csv_files.load_evaluation_not_processed(index_col=True)
    not_classified = list(set(x_dataset_not_processed.index.values.tolist()) - set(x_dataset_cat.index.values.tolist()))
    print(">> total time ", datetime.now() - start, "\n")
    return x_dataset_cat, not_classified


def input_test(url='http://127.0.0.1:5000/prediction', timeout=10):
    start, result = datetime.now(), []
    try:
        data_processed, not_classified = run_production()
        for e in not_classified: result.append([e, -1, -1])
        pickled_b64 = base64.b64encode(pickle.dumps(data_processed))
        response = requests.get(url, data=pickled_b64, timeout=timeout)
        response.raise_for_status()
        # result += response.json()
        print(response.json())
    except requests.exceptions.HTTPError as e:
        print(e)
    except requests.exceptions.Timeout as e:
        print(">> timeout exception: ", e)
    except requests.exceptions.TooManyRedirects as e:
        print(">> bad url found: ", e)
    except requests.exceptions.RequestException as e:
        print(">> bad request: ", e)
    print("\n>> total time ", datetime.now() - start, "\n")


if __name__ == "__main__":
    input_test()
