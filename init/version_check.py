#!/usr/bin/env python3

"""
@Author: Miro
@Date: 01/05/2020
@Objective: verifica delle librerie
@TODO:

******************************************************
Libraries
******************************************************
    1. Python3 --v 3.7.0
    2. Scikit-learn (Sklearn) --v 1.0.2
    3. TensorFlow (Keras) --v 2.9.1
    4. Pandas --v 1.1.5
    5. Numpy --v 1.21.6
    6. Matplotlib --v 3.5.2
******************************************************

"""

import tensorflow as tf
import sklearn as sk
import matplotlib
import numpy as np
import pandas as pd

print("TensorFlow version:", tf.__version__)
print("Sklearn version:", sk.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Numpy version:", np.__version__)
print("Pandas version:", pd.__version__)
