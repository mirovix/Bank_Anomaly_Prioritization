#!/usr/bin/env python3

"""
@Author: Miro
@Date: 25/10/2022
@Version: 1.0
@Objective: funzione di supporto per la produzione
@TODO:
"""

import base64
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from anomaly_info_extractor import concatenate_xml_predictions
from configs import production_db_config as pdbc, production_config as pc
from predict_anomaly import load_model, predict_model

model, threshold_comp, threshold_day = load_model()


def set_view_index(code_bank, view_path=pc.view_path, index_path=pc.index_path):
    sql_index, sql_view = read_query(index_path).split(';'), read_query(view_path).split(';')
    try:
        for sql in sql_index: pc.engine_dwa_single.execute(sql)
        print(">> index completed\n")
    except Exception:
        print(">> index not computed or done previously")

    try:
        for i in range(4): pc.engine_dwa_single.execute(sql_view[i] % ("'" + str(code_bank) + "'"))
        for i in range(4, 6): pc.engine_dwa_single.execute(sql_view[i])
        pc.engine_rx_input.execute(sql_view[6] + pc.rx_input_production_name)
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


def ndg_query_exe(ndg):
    ndg_query = ''
    for user_id in ndg: ndg_query += "'" + str(user_id) + "',"
    ndg_query = (' (' + ndg_query)[:-1] + ')'
    return ndg_query


def query_composition(querys_path, target_db_comp, target_db_day, target_db):
    data_op_comp, data_op_day, data_op_day_info = None, None, None

    sql_file = read_query(querys_path)
    sql_commands = sql_file.split(';')

    ndg = list(dict.fromkeys(target_db.NDG.values.tolist()[:pc.max_elements]))
    ndg_query = ndg_query_exe(ndg)

    sql_commands[0] += ndg_query
    sql_commands[1] += ndg_query

    if target_db_comp.shape[0] > 0:
        start_range_comp, end_range_comp, _ = range_date(target_db_comp.DATA.values.tolist()[:pc.max_elements_test])
        ndg_comp = list(dict.fromkeys(target_db_comp.NDG.values.tolist()[:pc.max_elements]))
        data_op_comp = sql_commands[2] + ndg_query_exe(ndg_comp) + query_date_range(start_range_comp, end_range_comp)
    if target_db_day.shape[0] > 0:
        start_range_day, end_range_day, start_range_day_info = range_date(target_db_day.DATA.values.tolist()[:pc.max_elements_test])
        ndg_day = list(dict.fromkeys(target_db_day.NDG.values.tolist()[:pc.max_elements]))
        data_op_day = sql_commands[2] + ndg_query_exe(ndg_day) + query_date_range(start_range_day, end_range_day)
        data_op_day_info = sql_commands[3] + ndg_query_exe(ndg_day) + query_date_range(start_range_day_info, end_range_day)

    return sql_commands, data_op_comp, data_op_day, data_op_day_info


def write_result(input_predictions, input_rx, query_output_name=pc.rx_output_production_name):
    try:
        input_predictions.replace(np.nan, None, inplace=True)
        for index, row in input_predictions.iterrows():
            input_rx.at[row.ID, pc.xml_col_name] = concatenate_xml_predictions(row.CONTENUTO,
                                                                               row.percentuale,
                                                                               row.fascia)
        input_rx = input_rx.reset_index(level=0)
        input_rx.to_sql(query_output_name, con=pc.engine_rx_output, if_exists='append', index=False)
    except Exception:
        print(">> error: something wrong happened in writing result")


def ws_prediction(data_processed, url=pdbc.web_service_url, timeout=pdbc.web_service_time_out):
    response = pd.DataFrame()

    try:
        pickled_b64 = base64.b64encode(pickle.dumps(data_processed))
        response = requests.get(url, data=pickled_b64, timeout=timeout)
        response.raise_for_status()

    except requests.exceptions.HTTPError as e:
        print(e)
    except requests.exceptions.Timeout as e:
        print(">> timeout exception: ", e)
    except requests.exceptions.TooManyRedirects as e:
        print(">> bad url found: ", e)
    except requests.exceptions.RequestException as e:
        print(">> bad request: ", e)

    return response


def predict(data_processed):
    if pc.web_service_flag is False:
        response = predict_model(data_processed, model, threshold_comp, threshold_day)
    else:
        response = ws_prediction(data_processed).json()

    # taking the maximum in case of duplicates
    df_response = pd.DataFrame(response, columns=pc.col_name).astype(pc.dtype_col) \
        .sort_values(pc.col_name, ascending=True) \
        .drop_duplicates(subset=[pc.index_name], keep="last")

    return df_response
