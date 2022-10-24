#!/usr/bin/env python3

"""
@Author: Miro
@Date: 07/10/2022
@Version: 1.0
@Objective: connesioni ai db
@TODO:
"""

from sqlalchemy.engine import URL
from configs import production_db_config as pdbc
from sqlalchemy import create_engine


def db_connection_dwa(sql_server_flag=False, my_sql_flag=False):
    info_db = pdbc.dwa_server, pdbc.dwa_database, pdbc.dwa_username, \
              pdbc.dwa_password, pdbc.sql_server_driver, pdbc.dwa_port
    return db_connection(info_db, sql_server_flag, my_sql_flag)


def db_connection_rx_input(sql_server_flag=False, my_sql_flag=False):
    info_db = pdbc.rx_server, pdbc.rx_database, pdbc.rx_username, \
              pdbc.rx_password, pdbc.sql_server_driver, pdbc.rx_port
    return db_connection(info_db, sql_server_flag, my_sql_flag)


def db_connection_rx_output(sql_server_flag=False, my_sql_flag=False):
    info_db = pdbc.rx_evaluated_server, pdbc.rx_evaluated_database, pdbc.rx_evaluated_username, \
              pdbc.rx_evaluated_password, pdbc.sql_server_driver, pdbc.rx_evaluated_port
    return db_connection(info_db, sql_server_flag, my_sql_flag)


def db_connection(info_db, sql_server_flag, my_sql_flag):
    engine, connection_url, connection_string = None, None, ''
    server, database, user, pw, driver, port = info_db
    if sql_server_flag is True:
        connection_string = driver + ';SERVER=' + server + ';PORT=' + port + ';DATABASE=' + database + ';UID=' + user + ';PWD=' + pw + ';TDS_Version=8.0'
        connection_url = URL.create(pdbc.sql_server_driver_name,
                                    query={pdbc.sql_server_query_driver: connection_string})
    elif my_sql_flag is True:
        connection_url = pdbc.mysql_driver_name + "://{0}:{1}@{2}:{3}/{4}".format(user, pw, server, port, database)
        connection_string = connection_url
    try:
        engine = create_engine(connection_url)
        print(">> connected to the db >> " + connection_string + "\n")
    except Exception as ex:
        print(">> connection could not be made due to the following error: ", ex)

    return engine
