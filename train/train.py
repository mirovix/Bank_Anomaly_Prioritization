#!/usr/bin/env python3

"""
@Author: Miro
@Date: 09/06/2022
@Version: 1.0
@Objective: training loop per allenare il modello
@TODO:
"""


import numpy as np
from datetime import datetime
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN
from categorization import Categorization
from functions_plot import *


def smote_over_sampling(data, target, smote_model):
    x_resample, y_resample = smote_model.fit_resample(np.array(data.values.tolist()), np.array(target.values.tolist()))
    target = pd.DataFrame(y_resample, columns=target.columns)
    data = pd.DataFrame(x_resample, columns=data.columns)
    return data, target


def plots_performance(classifier, x_test, y_test, threshold_comp, threshold_day, name='RANDOM_FOREST', pos_label=None):
    if pos_label is None:
        pos_label = [1, 3]

    y_pred_perc = classifier.predict_proba(x_test)
    y_pred = from_threshold_to_pred(y_pred_perc, threshold_comp, threshold_day)
    y_test_arr = np.array(y_test).reshape(len(y_test), )

    print(">> model acc. ", accuracy_score(y_test_arr, y_pred))

    cnf_matrix = confusion_matrix(y_test_arr, y_pred)
    plot_confusion_matrix(cnf_matrix, classes=[0, 1, 2, 3])
    plt.show()

    print(">> GENERAL EVALUATION", str(precision_recall_fscore_support(y_test_arr, y_pred)))

    y_pred_temp = np.delete(y_pred, np.where(y_pred > 1))
    y_test_arr_temp = np.delete(y_test_arr, np.where(y_test_arr > 1))
    plot_roc(y_pred_temp, y_test_arr_temp, pos_label[0])
    y_pred_temp = np.delete(y_pred, np.where(y_pred < 2))
    y_test_arr_temp = np.delete(y_test_arr, np.where(y_test_arr < 2))
    plot_roc(y_pred_temp, y_test_arr_temp, pos_label[1])

    plot_feature_importance(classifier.feature_importances_, x_test.columns, name)

    return y_pred_perc, y_test_arr


def from_threshold_to_pred(y_pred, threshold_comp, threshold_day):
    y_pred_list = []
    for e in y_pred.tolist():
        current_e_index = e.index(max(e))
        if current_e_index < 2:
            if e[1] >= threshold_comp:
                y_pred_list.append(1)
            else:
                y_pred_list.append(0)
        else:
            if e[3] >= threshold_day:
                y_pred_list.append(3)
            else:
                y_pred_list.append(2)
    return np.array(y_pred_list)


def threshold_check(x_val, y_val, model, step=0.01):
    f1_d_th, f1_c_th, best_th_day, best_th_comportamenti = 0, 0, 0, 0
    y_test = np.array(y_val).reshape(len(y_val), )
    for threshold in np.arange(0.05, 1, step):
        y_pred = model.predict_proba(x_val)
        y_pred = from_threshold_to_pred(y_pred, threshold, threshold)
        f1 = f1_score(y_test, y_pred, average=None).tolist()
        if f1[1] >= f1_c_th:
            f1_c_th = f1[1]
            best_th_comportamenti = threshold
        if f1[3] >= f1_d_th:
            f1_d_th = f1[3]
            best_th_day = threshold
    return best_th_comportamenti, best_th_day


def single_anomaly_comparison(y_model, y_target, threshold_comp, threshold_day, min_th_rmv_comp=0.02,
                              min_th_rmv_day=0.04):
    to_remove_comp, to_remove_day, to_remove_err_day, to_remove_err_comportamenti = 0, 0, 0, 0
    for i in range(y_target.shape[0]):
        if y_model[i][1] < min_th_rmv_comp and y_model[i][3] < 0.0001:
            to_remove_comp += 1
        elif y_model[i][3] < min_th_rmv_day and y_model[i][1] < 0.0001:
            to_remove_day += 1

        if y_model[i][1] > threshold_comp and y_target[i] == 0:
            print(" >> false positive COMPORTAMENTI >> y_model " + str(y_model[i]) + " y_target " + str(y_target[i]))
        elif y_model[i][1] < threshold_comp and y_target[i] == 1:
            print(" >> false NEGATIVE COMPORTAMENTI >> y_model " + str(y_model[i]) + " y_target " + str(y_target[i]))
            if y_model[i][1] < min_th_rmv_comp and y_model[i][3] < 0.0001:
                to_remove_err_comportamenti += 1
        elif y_model[i][3] > threshold_day and y_target[i] == 2:
            print(" >> false positive DAY >> y_model " + str(y_model[i]) + " y_target " + str(y_target[i]))
        elif y_model[i][3] < threshold_day and y_target[i] == 3:
            print(" >> false NEGATIVE DAY >> y_model " + str(y_model[i]) + " y_target " + str(y_target[i]))
            if y_model[i][3] < min_th_rmv_day and y_model[i][1] < 0.0001:
                to_remove_err_day += 1

    print(" >> remove COMPORTAMENTI threshold " + str(min_th_rmv_comp) + " with number of elements " + str(
        to_remove_comp))
    print(" >> remove DAY threshold " + str(min_th_rmv_day) + " with number of elements " + str(to_remove_day))
    print(" >> remove COMPORTAMENTI+DAY with number of elements " + str((to_remove_day + to_remove_comp)))
    print(" >> remove errors COMPORTAMENTI with number of elements " + str(to_remove_err_comportamenti))
    print(" >> remove errors DAY with number of elements " + str(to_remove_err_day))


def random_grid():
    n_estimators = [int(x) for x in np.linspace(start=1000, stop=16000, num=20)]

    class_weight = ['balanced', None]

    max_depth = [int(x) for x in np.linspace(30, 140, num=30)]
    max_depth.append(None)

    min_samples_split = [2, 3, 4, 5, 7]

    min_samples_leaf = [2, 3, 4, 5, 7]

    random_grid = {'n_estimators': n_estimators,
                   'class_weight': class_weight,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}

    print(">> parameters to search ", random_grid)
    return random_grid


def random_search_training(x, y, score='f1', n_iter=100, cv=3, ver=1, n_jobs=-1):
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid(), scoring=score, n_iter=n_iter, cv=cv,
                                   verbose=ver, n_jobs=n_jobs, return_train_score=True)
    rf_random.fit(x, y.values.ravel())
    print(">> best parameters ", rf_random.best_params_)
    return rf_random


def split_data(data, target, size_train):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=(1 - size_train), shuffle=True)
    return x_train, x_test, y_train, y_test


def run():
    start = datetime.now()
    data, target, data_anomaly_list, data_eval = Categorization().run(cat_num=True)

    x_train_orig, x_test_val, y_train_orig, y_test_val = split_data(data, target, size_train=0.75)
    x_test, x_val, y_test, y_val = split_data(x_test_val, y_test_val, size_train=0.65)

    print(">> size train ", x_train_orig.shape[0])
    print(">> size validation ", x_val.shape[0])
    print(">> size test ", x_test.shape[0])

    n_jobs = -1
    smote_models = [None, SMOTE(n_jobs=n_jobs, k_neighbors=1), SMOTE(n_jobs=n_jobs, k_neighbors=2),
                    SMOTE(n_jobs=n_jobs, k_neighbors=3), SMOTE(n_jobs=n_jobs, k_neighbors=4),
                    SMOTEENN(n_jobs=n_jobs), SMOTETomek(n_jobs=n_jobs),
                    ADASYN(n_jobs=n_jobs)]

    for i in range(len(smote_models)):
        if i > 0:
            print(">> smote ", str(i))
            x_train, y_train = smote_over_sampling(x_train_orig, y_train_orig, smote_models[i])
        else:
            x_train, y_train = x_train_orig, y_train_orig

        # model_search = random_search_training(x_train, y_train)
        # model = model_search.best_estimator_
        model = RandomForestClassifier(n_estimators=2000, verbose=1, n_jobs=7, min_samples_split=2,
                                       min_samples_leaf=2)  # , class_weight='balanced')

        # model = GradientBoostingClassifier(n_estimators=3500, verbose=1)

        model.fit(x_train, y_train.values.ravel())

        # filename = 'train/models/rfc_model_best_8000_4.sav'
        # pickle.dump(model, open(filename, 'wb'))

        threshold_comp, threshold_day = threshold_check(x_val, y_val, model)
        print(">> threshold comportamenti", threshold_comp)
        print(">> threshold day", threshold_day)

        y_model, y_target = plots_performance(model, x_test, y_test, threshold_comp, threshold_day)
        single_anomaly_comparison(y_model, y_target, threshold_comp, threshold_day)
    print(">> total time ", datetime.now() - start, "\n")


run()
