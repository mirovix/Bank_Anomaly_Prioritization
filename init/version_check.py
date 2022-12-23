#!/usr/bin/env python3

"""
@Author: Miro
@Date: 01/05/2020
@Objective: verifica delle librerie
@TODO:

******************************************************
Libraries
******************************************************
    1. Python3 --v 3.9.12
    2. TensorFlow (Keras) --v 2.9.1
    3. Scikit-learn (Sklearn) --v 1.1.1
    4. Pandas --v 1.4.2
    5. Numpy --v 1.22.4
    6. Matplotlib --v 3.5.2
    7. Flask --v 2.1.3
    8. Imblearn --v 0.9.1
    9. Missingno --v 0.5.1
    10. Sqlalchemy --v 1.4.39
******************************************************

"""

import sys
import tensorflow as tf
import sklearn as sk
import matplotlib
import numpy as np
import pandas as pd
import flask
import imblearn as imb
import missingno as msn
import sqlalchemy

if __name__ == "__main__":
    print("\n>> Python version: ", sys.version)
    print(">> TensorFlow version:", tf.__version__)
    print(">> Sklearn version:", sk.__version__)
    print(">> Matplotlib version:", matplotlib.__version__)
    print(">> Numpy version:", np.__version__)
    print(">> Pandas version:", pd.__version__)
    print(">> Flask version:", flask.__version__)
    print(">> Imblearn version:", imb.__version__)
    print(">> Missingno version:", msn.__version__)
    print(">> Sqlalchemy version:", sqlalchemy.__version__)
