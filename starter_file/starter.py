"""
@Author: Miro
@Date: 31/10/2022
@Version: 1.0
@Objective: file per iniziare la prioritization delle anomalie
@TODO:
"""

import sys

sys.path.extend(['C:\\workspace\\AnomalyPrioritization',
                 'C:\\workspace\\AnomalyPrioritization\\configs',
                 'C:\\workspace\\AnomalyPrioritization\\data',
                 'C:\\workspace\\AnomalyPrioritization\\init',
                 'C:\\workspace\\AnomalyPrioritization\\productions',
                 'C:\\workspace\\AnomalyPrioritization\\documentations_ML',
                 'C:\\workspace\\AnomalyPrioritization\\test',
                 'C:\\workspace\\AnomalyPrioritization\\input\\build_features_dir',
                 'C:\\workspace\\AnomalyPrioritization\\querys_DB',
                 'C:\\workspace\\AnomalyPrioritization\\train',
                 'C:\\workspace\\AnomalyPrioritization\\functions_plot',
                 'C:\\workspace\\AnomalyPrioritization\\input',
                 'C:\\workspace\\AnomalyPrioritization\\pre_processing_features',
                 'C:/workspace/AnomalyPrioritization'])

production_file_path = "C:/workspace/AnomalyPrioritization/productions/production.py"

try:
    exec(open(production_file_path).read())
except Exception as ex:
    print(">> error loading production file\n")
    exit(1)
