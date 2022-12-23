"""
@Author: Miro
@Date: 31/10/2022
@Version: 1.1
@Objective: configuration file per la connessione ai df locale
@TODO:
"""

from productions.pwd_encryption import decryption, read_key
from configs import production_config as pc

key = read_key()

# db types >> oracle, sqlserver, mysql

oracle_name = 'oracle'
sqlserver_name = 'sqlserver'
mysql_name = 'mysql'

# drivers

sql_server_driver = 'DRIVER=SQL Server Native Client 11.0'
sql_server_driver_name = 'mssql+pyodbc'
sql_server_query_driver = 'odbc_connect'

mysql_driver_name = 'mysql+pymysql'

oracle_driver_name = 'oracle+cx_oracle'
oracle_service = 'netech'
oracle_instant_client_path = pc.base_path + "/driver/instantclient_21_7"

# discovery day db

dwa_day_server_type = sqlserver_name
dwa_day_server = 'ub04'
dwa_day_database = 'mm_dwa'
dwa_day_username = 'sa'
dwa_day_password = decryption(pc.pwd_dwa_day_path, key)
dwa_day_port = '1433'

# discovery comportamenti db

dwa_comp_server_type = sqlserver_name
dwa_comp_server = 'ub04'
dwa_comp_database = 'mm_dwa'
dwa_comp_username = 'sa'
dwa_comp_password = decryption(pc.pwd_dwa_comp_path, key)
dwa_comp_port = '1433'

# rx input db

rx_server_type = sqlserver_name
rx_server = 'ub04'
rx_database = 'mm_dwa'
rx_username = 'sa'
rx_password = decryption(pc.pwd_rx_input_path, key)
rx_port = '1433'

# rx output db

rx_evaluated_server_type = sqlserver_name
rx_evaluated_server = 'ub04'
rx_evaluated_database = 'mm_dwa'
rx_evaluated_username = 'sa'
rx_evaluated_password = decryption(pc.pwd_rx_output_path, key)
rx_evaluated_port = '1433'

# web service configuration

web_service_app_route = '/prediction'

web_service_url = 'http://127.0.0.1:5000' + web_service_app_route
web_service_time_out = 10
