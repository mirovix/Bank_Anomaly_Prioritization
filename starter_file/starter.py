#!/usr/bin/env python3

"""
@Author: Miro
@Date: 31/10/2022
@Version: 1.0
@Objective: file per iniziare la prioritization delle anomalie
@TODO:
"""

import sys
sys.path.extend(['C:\\workspace\\AnomalyPrioritization'])
from extend_modules import extend_modules
extend_modules()

production_file_path = "C:/workspace/AnomalyPrioritization/productions/production.py"

try:
    exec(open(production_file_path).read())
except Exception as ex:
    print(">> error loading production file >> " + str(ex) + "\n")
    sys.exit(1)
