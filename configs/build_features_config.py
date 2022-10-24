# basic information for building the features

positive_target_comp, positive_target_day = 1, 3
negative_target_comp, negative_target_day = 0, 2
name_comp, name_day = 'comportamenti', 'day'

size_train = 0.8
size_test = 0.35

path_x = "C:/workspace/AnomalyPrioritization/data/dataset/dataset_x.csv"
path_y = "C:/workspace/AnomalyPrioritization/data/dataset/dataset_y.csv"
path_x_evaluated = "C:/workspace/AnomalyPrioritization/data/dataset/dataset_x_evaluated.csv"
path_x_train = "C:/workspace/AnomalyPrioritization/data/dataset_split/x_train.csv"
path_y_train = "C:/workspace/AnomalyPrioritization/data/dataset_split/y_train.csv"
path_x_val = "C:/workspace/AnomalyPrioritization/data/dataset_split/x_val.csv"
path_y_val = "C:/workspace/AnomalyPrioritization/data/dataset_split/y_val.csv"
path_x_test = "C:/workspace/AnomalyPrioritization/data/dataset_split/x_test.csv"
path_y_test = "C:/workspace/AnomalyPrioritization/data/dataset_split/y_test.csv"

range_repetitiveness = 0.05
variance_threshold_1 = 0.80
variance_threshold_2 = 0.75
variance_threshold_filiale = 35
min_age = 11
max_age = 110
step_age = 10
causali_version = 14
months = [3, 6]
prefix = ["tot", "media", "num", "ripetitività_mov", "ripetitività_num"]

# data for building features

day_col_x_only = ["CAUSALE", "COUNTRY_OPERATION", "SIGN", "AMOUNT_OPERATION", "AMOUNT_OPERATION_CASH",
                  "COUNTRY_CONTROPARTE_OPERATION", "OPERATION_COUNTRY_EXECUTOR", "RISK_PROFILE_EXECUTOR"]
fill_with_string = ['AMOUNT_OPERATION', 'AMOUNT_OPERATION_CASH', 'RISK_PROFILE_EXECUTOR']

value_default_dd_dc = -1
name_default_dd_dc = "OTHER_DB"

# data for managing missing values

dtypes_num = ['NCHECKREQUIRED', 'NCHECKDEBITED', 'NCHECKAVAILABLE', 'RISK_PROFILE']
dtypes_cat = ['REPORTED']
y_col_name = "EVALUATION"

# data for handling the categorization

columns_encoded_name = 'columns_encoded'
columns_encoded_num_name = 'columns_encoded_num'
cat_col_name = 'cat_col'
num_col_name = 'num_col'
data_enc_name = 'encoder_cat.joblib'
data_enc_num_name = 'encoder_num.joblib'
