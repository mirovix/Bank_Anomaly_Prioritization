"""
@Author: Miro
@Date: 31/10/2022
@Version: 1.0
@Objective: configuration file per la connessione ai df locale
@TODO:
"""

sql_server_driver = 'DRIVER=SQL Server Native Client 11.0'
sql_server_driver_name = 'mssql+pyodbc'
sql_server_query_driver = 'odbc_connect'

mysql_driver_name = 'mysql+pymysql'

dwa_day_server = 'ub04'
dwa_day_database = 'mm_dwa'
dwa_day_username = 'sa'
dwa_day_password = 'Ma1al3305'
dwa_day_port = '1433'

dwa_comp_server = 'ub04'
dwa_comp_database = 'mm_dwa'
dwa_comp_username = 'sa'
dwa_comp_password = 'Ma1al3305'
dwa_comp_port = '1433'

rx_server = 'ub04'
rx_database = 'mm_dwa'
rx_username = 'sa'
rx_password = 'Ma1al3305'
rx_port = '1433'

rx_evaluated_server = 'ub04'
rx_evaluated_database = 'mm_dwa'
rx_evaluated_username = 'sa'
rx_evaluated_password = 'Ma1al3305'
rx_evaluated_port = '1433'

web_service_app_route = '/prediction'

web_service_url = 'http://127.0.0.1:5000' + web_service_app_route
web_service_time_out = 10
