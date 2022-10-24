import numpy as np

index_name = 'ID'
code_bank = '060459'
len_ndg = 16
testing_flag = False
time_to_sleep = 60 * 60

# path directories

query_production_path = 'C:/workspace/AnomalyPrioritization/querys_DB/query/production_ML.sql'
rx_query_production_path = 'C:/workspace/AnomalyPrioritization/querys_DB/query/rx_production_ML.sql'
rx_input_production_name = 'RX_INPUT'
rx_output_production_name = 'RX_OUTPUT'
view_path = 'C:/workspace/AnomalyPrioritization/querys_DB/view/view_ML.sql'
index_path = 'C:/workspace/AnomalyPrioritization/querys_DB/index/index.sql'

machine_learning_model_path = "C:/workspace/AnomalyPrioritization/train/model_data/voting_model.sav"
machine_learning_categorization_data_path = "C:/workspace/AnomalyPrioritization/train/model_data/"
machine_learning_thresholds_data_path = 'C:/workspace/AnomalyPrioritization/train/model_data/'

# call web service data

col_name = ['ID', 'percentuale', 'fascia']
dtype_col = {col_name[0]: np.int64, col_name[1]: np.float64, col_name[2]: np.int8}
output_rows = ['SYSTEM', 'ID_TRASMISSIONE', 'ID', 'TIMESTAMP', 'CONTENUTO', 'DESTINATARIO']
xml_col_name = 'CONTENUTO'
data_operation_col_name = 'DATE_OPERATION'
not_predicted_value_perc, not_predicted_value_fascia = -1, -1

max_elements = None
engine_dwa = None
engine_rx_input = None
engine_rx_output = None

# anomaly information extractor data

xml_names = {'id': 'ID', 'op_date': 'DATA_OPERATION', 'ndg': 'NDG', 'importo': 'IMPORTO',
             'cod_anomalia': 'CODICE_ANOMALIA', 'sw': 'SOFTWARE', 'cod_op': 'A03',
             'data_anomalia': 'DATA', 'stato': 'STATO'}

dict_target = {'amount': xml_names['importo'], 'NDG': xml_names['ndg'],
               'operationDate': xml_names['op_date'], 'operationCode': xml_names['cod_op']}

dict_target_comp = {'NDG': xml_names['ndg']}
iso_date_len = 27
name_table_input = 'input_ML'
system_comp_name = 'COMPORTAMENT'
system_day_name = 'DISCOVERY'

# data for testing anomaly prediction

path_testing_file = "C:/workspace/AnomalyPrioritization/data/dataset_test/test.csv"
