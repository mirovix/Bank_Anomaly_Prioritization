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
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from configs import production_db_config as pdbc, production_config as pc
from productions.predict_anomaly import load_model, predict_model

model, threshold_comp, threshold_day = load_model()


def set_view_index(code_bank, engine, view_path=pc.view_path, index_path=pc.index_path):
    sql_index, sql_view = read_query(index_path).split(';'), read_query(view_path).split(';')
    try:
        for sql in sql_index: engine.execute(sql)
        print(">> index completed\n")
    except Exception:
        print(">> index not computed or done previously")

    try:
        for sql in sql_view: engine.execute(sql % ("'" + str(code_bank) + "'"))
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
    return start_range, end_range


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


def query_composition(querys_path, target_db):
    sql_file = read_query(querys_path)
    sql_commands = sql_file.split(';')

    ndg = list(dict.fromkeys(target_db.NDG.values.tolist()[:pc.max_elements]))
    ndg_query = ndg_query_exe(ndg)
    start_range, end_range = range_date(target_db.DATA.values.tolist())
    date_query = query_date_range(start_range, end_range)

    sql_commands[0] += ndg_query + '))'
    sql_commands[1] += ndg_query + ')'
    sql_commands[2] += ndg_query + date_query
    sql_commands[3] += ndg_query + ')' + date_query

    return sql_commands


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
