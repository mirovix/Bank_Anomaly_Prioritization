#!/usr/bin/env python3

"""
@Author: Miro
@Date: 30/05/2022
@Version: 1.13
@Objective: caricamento dei csv, creazione delle features e salvataggio nei file.
@TODO:
"""

from load_data import LoadData
from build_features_dir.build_features import BuildFeatures
from datetime import datetime
from load_dataset_split import split_dataset

if __name__ == "__main__":
    max_rows = 50
    start = datetime.now()
    features = BuildFeatures(LoadData(), max_elements=max_rows)
    features.get_dataset()
    features.save_dataset_csv()
    split_dataset()
    print(">> total time ", datetime.now() - start, "\n")
