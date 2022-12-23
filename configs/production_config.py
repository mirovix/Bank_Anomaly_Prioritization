"""
@Author: Miro
@Date: 26/10/2022
@Version: 1.0
@Objective: configuration file for production
@TODO:
"""

import numpy as np

# base configuration

index_name = 'ID'
code_bank = '060459'
len_ndg = 16

testing_flag = True
max_elements_test = 250
time_to_sleep_test = 30  # [s]

verbose = 1
time_to_sleep = 60 * 60  # [s]
web_service_flag = False
start_as_service = True

rx_input_production_name = 'RX_INPUT'
rx_output_production_name = 'RX_OUTPUT'
base_path = 'C:/workspace/AnomalyPrioritization'

# path directories

query_production_path = base_path + '/queries/query/production.sql'
rx_query_production_path = base_path + '/queries/query/rx_production.sql'
view_path = base_path + '/queries/view/view.sql'
index_path = base_path + '/queries/index/index.sql'

machine_learning_model_path = base_path + "/model_data/voting_model.sav"
machine_learning_categorization_data_path = base_path + "/model_data/"
machine_learning_thresholds_data_path = base_path + '/model_data/'

pwd_dwa_comp_path = base_path + "/data/password_encrypt/pwd_dwa_comp.txt"
pwd_dwa_day_path = base_path + "/data/password_encrypt/pwd_dwa_day.txt"
pwd_rx_input_path = base_path + "/data/password_encrypt/pwd_rx_input.txt"
pwd_rx_output_path = base_path + "/data/password_encrypt/pwd_rx_output.txt"
pwd_key_path = base_path + "/data/password_encrypt/key.txt"

# call web service data

col_name = ['ID', 'score', 'fascia']
dtype_col = {col_name[0]: np.int64, col_name[1]: np.float64, col_name[2]: np.int8}
output_rows_name = ['SYSTEM', 'ID_TRASMISSIONE', 'ID', 'TIMESTAMP', 'CONTENUTO', 'DESTINATARIO']
input_rows_name = ['SYSTEM', 'ID_TRASMISSIONE', 'ID', 'TIMESTAMP', 'CONTENUTO', 'DESTINATARIO']
xml_col_name = 'CONTENUTO'
data_operation_col_name = 'DATE_OPERATION'
not_predicted_value_perc, not_predicted_value_fascia = -1, 9

max_elements = max_elements_test
engine_rx_input = None
engine_rx_output = None
engine_dwa_day = None
engine_dwa_comportamenti = None

# anomaly information extractor data

xml_names = {'id': 'ID', 'op_date': 'DATA_OPERATION', 'ndg': 'NDG', 'importo': 'IMPORTO',
             'cod_anomalia': 'CODICE_ANOMALIA', 'sw': 'SOFTWARE', 'cod_op': 'A03',
             'data_anomalia': 'DATA', 'stato': 'STATO'}

dict_target = {'amount': xml_names['importo'], 'NDG': xml_names['ndg'],
               'operationDate': xml_names['op_date'], 'operationCode': xml_names['cod_op']}

dict_target_comp = {'NDG': xml_names['ndg'], 'amount': xml_names['importo']}

iso_date_len = 27
name_table_input = rx_input_production_name
system_comp_name = 'COMPORTAMENT'
system_day_name = 'DISCOVERY'
