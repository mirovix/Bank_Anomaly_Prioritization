#!/usr/bin/env python3

"""
@Author: Miro
@Date: 07/10/2022
@Version: 1.1
@Objective: connesioni ai db
@TODO:
"""

from sqlalchemy.engine import URL
from configs import production_db_config as pdbc
from sqlalchemy import create_engine
from sys import exit
import cx_Oracle

try:
    cx_Oracle.init_oracle_client(lib_dir=pdbc.oracle_instant_client_path)
except Exception as ex:
    print("\n>> Oracle instant client not found\n")


def db_connection_dwa_day():
    info_db = pdbc.dwa_day_server, pdbc.dwa_day_database, pdbc.dwa_day_username, \
              pdbc.dwa_day_password, pdbc.sql_server_driver, pdbc.dwa_day_port
    return db_connection(info_db, pdbc.dwa_day_server_type)


def db_connection_dwa_comp():
    info_db = pdbc.dwa_comp_server, pdbc.dwa_comp_database, pdbc.dwa_comp_username, \
              pdbc.dwa_comp_password, pdbc.sql_server_driver, pdbc.dwa_comp_port
    return db_connection(info_db, pdbc.dwa_comp_server_type)


def db_connection_rx_input():
    info_db = pdbc.rx_server, pdbc.rx_database, pdbc.rx_username, \
              pdbc.rx_password, pdbc.sql_server_driver, pdbc.rx_port
    return db_connection(info_db, pdbc.rx_server_type)


def db_connection_rx_output():
    info_db = pdbc.rx_evaluated_server, pdbc.rx_evaluated_database, pdbc.rx_evaluated_username, \
              pdbc.rx_evaluated_password, pdbc.sql_server_driver, pdbc.rx_evaluated_port
    return db_connection(info_db, pdbc.rx_evaluated_server_type)


def db_connection(info_db, server_type):
    engine, connection_url, connection_string = None, None, ''
    server, database, user, pw, driver, port = info_db
    if server_type == pdbc.sqlserver_name:
        connection_string = driver + ';SERVER=' + server + ';PORT=' + port + ';DATABASE=' + database + ';UID=' + user + ';PWD=' + pw + ';TDS_Version=8.0'
        connection_url = URL.create(pdbc.sql_server_driver_name,
                                    query={pdbc.sql_server_query_driver: connection_string})
    elif server_type == pdbc.mysql_name:
        connection_url = pdbc.mysql_driver_name + "://{0}:{1}@{2}:{3}/{4}".format(user, pw, server, port, database)

    elif server_type == pdbc.oracle_name:
        connection_url = pdbc.oracle_driver_name + "://{0}:{1}@{2}:{3}/?service_name={4}".format(user, pw, server, port, pdbc.oracle_service)
    try:
        engine = create_engine(connection_url)
        print(">> connection db" +
              " >> SERVER: " + server +
              " >> DATABASE: " + database +
              " >> PORT: " + port +
              " >> USER: " + user + "\n")
    except Exception as ex:
        print(">> connection could not be made due to the following error: ", ex)
        exit(1)

    return engine
