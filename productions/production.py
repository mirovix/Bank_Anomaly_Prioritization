#!/usr/bin/env python3

"""
@Author: Miro
@Date: 12/09/2022
@Version: 1.2
@Objective: input test del web service
@TODO:
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd
from configs import production_config as pc
from build_features_dir.build_features import BuildFeatures
from categorization import Categorization
from generation_fake_xml import generation_new_data
from db_connection import db_connection_dwa_comp, db_connection_dwa_day, db_connection_rx_input, db_connection_rx_output
from anomaly_info_extractor import build_target
from load_data import LoadData
from util_production import read_query, query_composition, set_view_index, predict, write_result

load_data = LoadData()


def run_production(querys_path=pc.query_production_path,
                   rx_query_path=pc.rx_query_production_path):
    target_db_comp, target_db_day, input_rx = build_target(read_query(rx_query_path), pc.engine_rx_input, max_elements=pc.max_elements)
    if input_rx is None: return None, None, None
    target_db = pd.concat([target_db_comp, target_db_day])

    sql, data_query_comp, data_query_day, data_query_day_info = query_composition(querys_path, target_db_comp, target_db_day, target_db)

    subject_db = (sql[0], pc.engine_dwa_day)
    account_db = (sql[1], pc.engine_dwa_comportamenti)
    operations_db = (data_query_comp, pc.engine_dwa_comportamenti,
                     data_query_day, pc.engine_dwa_day)
    operations_day_db = (data_query_day_info, pc.engine_dwa_day)

    sys.stdout.write(">> loading data from database...\r")
    features = BuildFeatures(load_data, production=True, max_elements=pc.max_elements,
                             subject_db=subject_db, target_db=target_db, account_db=account_db,
                             operations_db=operations_db, operations_day_db=operations_day_db)

    x_dataset_cat = Categorization(features.get_dataset()).run_production()
    target_db.set_index(pc.index_name, inplace=True)
    not_class = list(set(target_db.index.values.tolist()[:pc.max_elements]) - set(x_dataset_cat.index.values.tolist()))
    return x_dataset_cat, not_class, input_rx


def execute():
    start, result, df_response = datetime.now(), [], pd.DataFrame()

    data_processed, not_classified, input_rx = run_production()

    if data_processed is None:
        return
    elif data_processed.shape[0] > 0:
        df_response = predict(data_processed)

    for e in not_classified:
        result.append([e, pc.not_predicted_value_perc, pc.not_predicted_value_fascia])
    df_response_not_class = pd.DataFrame(result, columns=pc.col_name).astype(pc.dtype_col)

    write_result(pd.merge(pd.concat([df_response, df_response_not_class]), input_rx, on=pc.index_name, how='left'),
                 input_rx)

    print("\n>> prediction done in ", datetime.now() - start, "\n")


def testing():
    if pc.start_as_service is False:
        generation_new_data(pc.engine_rx_input, pc.max_elements_test)
        execute()
        exit(0)

    i = 0
    while True:
        if i % 5 == 0: generation_new_data(pc.engine_rx_input, pc.max_elements_test)
        execute()
        i += 1
        time.sleep(pc.time_to_sleep_test)
        print(">> checking if new data are available\n")


if __name__ == "__main__":
    if pc.verbose == 0: sys.stdout = open(os.devnull, 'w')

    print("\n>> system started \n")
    sql_server_flag = True

    pc.engine_dwa_comportamenti = db_connection_dwa_comp(sql_server_flag=sql_server_flag)
    pc.engine_dwa_day = db_connection_dwa_day(sql_server_flag=sql_server_flag)
    pc.engine_rx_input = db_connection_rx_input(sql_server_flag=sql_server_flag)
    pc.engine_rx_output = db_connection_rx_output(sql_server_flag=sql_server_flag)
    set_view_index(code_bank=pc.code_bank)

    if pc.testing_flag is True: testing()
    pc.max_elements = None

    if pc.start_as_service is False:
        execute()
        exit(0)

    while True:
        execute()
        time.sleep(pc.time_to_sleep)
        print(">> checking if new data are available\n")
