#!/usr/bin/env python3

"""
@Author: Miro
@Date: 12/09/2022
@Version: 1.2
@Objective: input test del web service
@TODO:
"""

import pickle
import base64
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from configs import production_db_config as pdbc, production_config as pc
from build_features_dir.build_features import BuildFeatures
from categorization import Categorization
from generation_fake_xml import generation_new_data
from db_connection import db_connection_dwa, db_connection_rx_input, db_connection_rx_output
from anomaly_info_extractor import build_target, concatenate_xml_predictions
from load_data import LoadData

load_data = LoadData()


def set_view_index(code_bank, view_path=pc.view_path, index_path=pc.index_path):
    sql_index, sql_view = read_query(index_path).split(';'), read_query(view_path).split(';')
    try:
        for sql in sql_index: engine_dwa.execute(sql)
        print(">> index completed\n")
    except Exception:
        print(">> index not computed or done previously")

    try:
        for i in range(4): engine_dwa.execute(sql_view[i] % ("'" + str(code_bank) + "'"))
        for i in range(4, 6): engine_dwa.execute(sql_view[i])
        engine_rx_input.execute(sql_view[6] + pc.rx_input_production_name)
        print(">> view completed\n")
    except Exception:
        print(">> view not computed or done previously\n")


def range_date(data):
    data_anomaly = datetime.fromisoformat(max(data))
    end_range = data_anomaly - relativedelta(hours=data_anomaly.hour - 24, minutes=data_anomaly.minute,
                                             seconds=data_anomaly.second,
                                             microseconds=data_anomaly.microsecond)
    data_anomaly = datetime.fromisoformat(min(data))
    start_range = data_anomaly - relativedelta(months=8, days=data_anomaly.day - 1,
                                               hours=data_anomaly.hour, minutes=data_anomaly.minute,
                                               seconds=data_anomaly.second,
                                               microseconds=data_anomaly.microsecond)
    start_range_day = data_anomaly - relativedelta(days=6,
                                                   hours=data_anomaly.hour, minutes=data_anomaly.minute,
                                                   seconds=data_anomaly.second,
                                                   microseconds=data_anomaly.microsecond)
    return start_range, end_range, start_range_day


def read_query(path):
    fd = open(path, 'r')
    sql_file = fd.read()
    fd.close()
    return sql_file


def query_date_range(start, end):
    return " and " + pc.data_operation_col_name + " >= '" + str(start) + \
           "' and " + pc.data_operation_col_name + " <= '" + str(end) + "'"


def query_composition(querys_path, start_range, end_range, start_range_day, ndg):
    sql_file = read_query(querys_path)
    sql_commands = sql_file.split(';')

    ndg_query = ''
    for user_id in ndg: ndg_query += "'" + str(user_id) + "',"
    ndg_query = (' (' + ndg_query)[:-1] + ')'
    data_query_op = query_date_range(start_range, end_range)
    data_query_op_day = query_date_range(start_range_day, end_range)

    for i in range(len(sql_commands)): sql_commands[i] += ndg_query
    return sql_commands, data_query_op, data_query_op_day


def run_production(max_elements=pc.max_elements,
                   querys_path=pc.query_production_path,
                   rx_query_path=pc.rx_query_production_path):
    target_db, input_rx = build_target(read_query(rx_query_path), engine_rx_input, max_elements=max_elements)
    if input_rx is None:
        return None, None, None
    ndg = list(dict.fromkeys(target_db.NDG.values.tolist()[:max_elements]))
    start_range, end_range, start_range_day = range_date(target_db.DATA.values.tolist()[:max_elements])

    start = datetime.now()

    sql, data_query, data_query_day = query_composition(querys_path, start_range, end_range, start_range_day, ndg)

    subject_db = (sql[0], engine_dwa)
    account_db = (sql[1], engine_dwa)
    operations_db = (sql[2] + data_query, engine_dwa)
    operations_day_db = (sql[3] + data_query_day, engine_dwa)

    sys.stdout.write(">> loading data from the database... take a break, it'll require a lot of time :) \r")
    features = BuildFeatures(load_data, production=True, max_elements=max_elements,
                             subject_db=subject_db, target_db=target_db, account_db=account_db,
                             operations_db=operations_db, operations_day_db=operations_day_db)

    x_dataset_cat = Categorization(features.get_dataset()).run_production()
    target_db.set_index(pc.index_name, inplace=True)
    not_classified = list(
        set(target_db.index.values.tolist()[:max_elements]) - set(x_dataset_cat.index.values.tolist()))
    print(">> total time ", datetime.now() - start, "\n")
    return x_dataset_cat, not_classified, input_rx


def write_result(input_predictions, input_rx, query_output_name=pc.rx_output_production_name):
    try:
        input_predictions.replace(np.nan, None, inplace=True)
        for index, row in input_predictions.iterrows():
            input_rx.at[row.ID, pc.xml_col_name] = concatenate_xml_predictions(row.CONTENUTO,
                                                                               row.percentuale,
                                                                               row.fascia)
        input_rx = input_rx.reset_index(level=0)
        input_rx.to_sql(query_output_name, con=engine_rx_output, if_exists='append', index=False)
    except Exception:
        print(">> error: something from happened in writing result")


def input_test(url=pdbc.web_service_url, timeout=pdbc.web_service_time_out):
    start, result = datetime.now(), []
    try:
        data_processed, not_classified, input_rx = run_production()
        if data_processed is None:
            return
        for e in not_classified:
            result.append([e, pc.not_predicted_value_perc, pc.not_predicted_value_fascia])
        df_response_not_class, df_response = pd.DataFrame(result, columns=pc.col_name).astype(
            pc.dtype_col), pd.DataFrame()
        if data_processed.shape[0] > 0:
            pickled_b64 = base64.b64encode(pickle.dumps(data_processed))
            response = requests.get(url, data=pickled_b64, timeout=timeout)
            response.raise_for_status()

            # taking the maximum in case of duplicates
            df_response = pd.DataFrame(response.json(), columns=pc.col_name).astype(pc.dtype_col) \
                .sort_values(pc.col_name, ascending=True) \
                .drop_duplicates(subset=[pc.index_name], keep="last")
        write_result(pd.merge(
            pd.concat([df_response, df_response_not_class]), input_rx, on=pc.index_name, how='left'),
            input_rx)

    except requests.exceptions.HTTPError as e:
        print(e)
    except requests.exceptions.Timeout as e:
        print(">> timeout exception: ", e)
    except requests.exceptions.TooManyRedirects as e:
        print(">> bad url found: ", e)
    except requests.exceptions.RequestException as e:
        print(">> bad request: ", e)
    print("\n>> prediction done in ", datetime.now() - start, "\n")


def testing(max_elements=250, time_sleep=100):
    generation_new_data(engine_rx_input, max_elements)
    i = 0
    while True:
        input_test()
        i += 1
        if i % 5 == 0: generation_new_data(engine_rx_input, max_elements)
        time.sleep(time_sleep)


if __name__ == "__main__":
    print(">> system started \n")
    sql_server_flag = True
    pc.testing_flag = False

    engine_dwa = db_connection_dwa(sql_server_flag=sql_server_flag)
    engine_rx_input = db_connection_rx_input(sql_server_flag=sql_server_flag)
    engine_rx_output = db_connection_rx_output(sql_server_flag=sql_server_flag)
    set_view_index(code_bank=pc.code_bank)

    if pc.testing_flag is True: testing()

    while True:
        input_test()
        time.sleep(pc.time_to_sleep)
