#!/usr/bin/env python3

"""
@Author: Miro
@Date: 30/05/2022
@Version: 1.13
@Objective: caricamento dei csv, creazione delle features e salvataggio nei file.
@TODO:
"""

from load_csv import LoadData
from build_features import BuildFeatures
from datetime import datetime
from load_dataset_split import split_dataset


def run(max_rows=50):
    start = datetime.now()
    csv_files = LoadData(max_months_considered=26)
    features = BuildFeatures(csv_files, max_rows)
    x_dataset, y_dataset = features.get_dataset(discovery_day=True, discovery_comportamenti=True)
    features.save_dataset_csv()
    # split_dataset()
    print(">> total time ", datetime.now() - start, "\n")
    return x_dataset, y_dataset


if __name__ == "__main__":
    run()
