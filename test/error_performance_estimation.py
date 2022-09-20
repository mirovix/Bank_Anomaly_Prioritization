#!/usr/bin/env python3

"""
@Author: Miro
@Date: 07/09/2022
@Version: 1.0
@Objective: Analisi degli errori sul test
@TODO:
"""

import pickle
import pandas as pd
from load_dataset_split import load_dataset_split
from models_definition import *


def error_estimation(path_save="errors.csv", filename_model='../train/models/voting_model.sav'):
    x_test, y_test = load_dataset_split(path_x_test="../data/dataset_split/x_test.csv",
                                        path_y_test="../data/dataset_split/y_test.csv",
                                        test=True)

    soglie = ['Irrilevante', 'Bassa', 'Media', 'Alta']
    target_list = [[0, 1, 0.10], [2, 3, 0.26]]
    threshold = [0.20, 0.60]
    loaded_model = pickle.load(open(filename_model, 'rb'))

    y_pred = loaded_model.predict_proba(x_test)
    y_test = np.array(y_test).reshape(len(y_test), )

    index_list, errors = x_test.index.values.tolist(), []
    for i in range(y_test.shape[0]):
        for n_target, p_target, f1_th in target_list:
            if p_target == 1:
                name = "Comportamenti"
            else:
                name = "Day"

            if y_pred[i][p_target] > threshold[1] and y_test[i] == n_target:
                errors.append([index_list[i], y_pred[i][p_target]*100, "Da segnalare", soglie[3], "Non da segnalare", name])
            elif f1_th < y_pred[i][p_target] <= threshold[1] and y_test[i] == n_target:
                errors.append([index_list[i], y_pred[i][p_target]*100, "Da segnalare", soglie[2], "Non da segnalare", name])
            elif threshold[0] < y_pred[i][p_target] <= f1_th and y_test[i] == p_target:
                errors.append([index_list[i], y_pred[i][p_target]*100, "Non da segnalare", soglie[2], "Da segnalare", name])
            elif y_pred[i][p_target] <= threshold[0] and y_test[i] == p_target:
                errors.append([index_list[i], y_pred[i][p_target]*100, "Non da segnalare", soglie[1], "Da segnalare", name])
    df = pd.DataFrame(errors, columns=['ID', 'Predizione percentuale [%]', 'Predizione', 'Priorità', 'Effettivo', 'Software'])
    df.to_csv(path_save, index=False)
    print(">> performance of the errors are saved in "+path_save)


error_estimation()
