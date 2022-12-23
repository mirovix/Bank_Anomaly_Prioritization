#!/usr/bin/env python3

"""
@Author: Miro
@Date: 14/11/2022
@Version: 1.1
@Objective: estrattore di csv dai db
@TODO:
"""

import sys
import pandas as pd
from sqlalchemy.engine import URL
import config
from sqlalchemy import create_engine
from sys import exit


def read_queries():
    # use delimitar ';' for splitting multiple queries
    fd = open(config.queries_file, 'r')
    sql_commands = fd.read().split(';')
    fd.close()
    return sql_commands


def db_connection():
    engine, connection_url, connection_string = None, None, ''
    if config.sql_server_flag is True:
        connection_string = config.sql_server_driver + \
                            ';SERVER=' + config.sql_server_server + \
                            ';PORT=' + config.sql_server_port + \
                            ';DATABASE=' + config.sql_server_database + \
                            ';UID=' + config.sql_server_username + \
                            ';PWD=' + config.sql_server_password + \
                            ';TDS_Version=8.0'
        connection_url = URL.create(config.sql_server_driver_name,
                                    query={config.sql_server_query_driver: connection_string})
    elif config.mysql_flag is True:
        connection_url = config.mysql_driver_name + "://{0}:{1}@{2}:{3}/{4}".format(config.mysql_username,
                                                                                    config.mysql_password,
                                                                                    config.mysql_server,
                                                                                    config.mysql_port,
                                                                                    config.mysql_database)
        connection_string = connection_url
    elif config.oracle_flag is True:
        connection_url = config.oracle_driver_name + "://{0}:{1}@{2}:{3}/?service_name={4}".format(config.oracle_username,
                                                                                                   config.oracle_password,
                                                                                                   config.oracle_server,
                                                                                                   config.oracle_port,
                                                                                                   config.oracle_service)
        connection_string = connection_url
    try:
        engine = create_engine(connection_url)
        print(">> connection db >> " + connection_string + "\n")
    except Exception as ex:
        print(">> connection could not be made due to the following error: ", ex)
        exit(1)

    return engine


def extract():
    print("\n>> extractor system started\n")
    queries = read_queries()
    engine = db_connection()
    for i, query in enumerate(queries):
        try:
            pd.read_sql(query, engine).astype(str)\
              .to_csv(config.output_dir + '/query_' + str(i + 1) + '.csv', index=False)
            to_write = ">> csv completed " + str(i + 1) + "/" + str(len(queries)) + "\r"
            sys.stdout.write(to_write)
        except Exception as ex:
            print(">> error reading/saving data from db: ", ex)
            exit(0)

    print(">> extraction successfully completed")


if __name__ == "__main__":
    extract()
    exit(0)
