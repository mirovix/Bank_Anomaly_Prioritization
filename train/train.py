#!/usr/bin/env python3

"""
@Author: Miro
@Date: 09/06/2022
@Version: 1.0
@Objective: training loop per allenare il modello
@TODO: soglia automatica del medio livello
"""

import pickle
from datetime import datetime
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, make_scorer
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import LabelBinarizer
from functions_plot import *
from load_dataset_split import load_dataset_split
from models_definition import *


def smote_over_sampling(data, target, smote_model):
    x_resample, y_resample = smote_model.fit_resample(np.array(data.values.tolist()), np.array(target.values.tolist()))
    target = pd.DataFrame(y_resample, columns=target.columns)
    data = pd.DataFrame(x_resample, columns=data.columns)
    return data, target


def plots_performance(classifier, x_test, y_test, threshold_comp, threshold_day, all_threshold_comp, all_threshold_day,
                      pos_label=None):
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

    # plot_feature_importance(classifier.feature_importance_, x_test.columns, name='RANDOM_FOREST')

    plot_table_soglie(distribution_anomaly(y_pred_perc, y_test_arr, all_threshold_comp, threshold_comp))
    plot_table_soglie(distribution_anomaly(y_pred_perc, y_test_arr, all_threshold_day, threshold_day, True))

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


def my_custom_loss_func(y_true, y_pred):
    f1_c_th, f1_d_th = 0, 0
    for threshold in np.arange(0.05, 1, 0.01):
        y_pred_ = from_threshold_to_pred(y_pred, threshold, threshold)
        f1 = f1_score(y_true, y_pred_, average=None).tolist()
        if f1[1] >= f1_c_th:
            f1_c_th = f1[1]
        if f1[3] >= f1_d_th:
            f1_d_th = f1[3]
    return (f1_c_th + f1_d_th) / 2


def threshold_check(x_val, y_val, model, step=0.01, step_min_th=0.00001):
    f1_d_th, f1_c_th, best_th_day, best_th_comportamenti, best_min_th_day, best_min_th_comportamenti = 0, 0, 0, 0, 0, 0
    y_val_arr = np.array(y_val).reshape(len(y_val), )
    y_pred_perc = model.predict_proba(x_val)
    for threshold in np.arange(0.05, 1, step):
        y_pred = from_threshold_to_pred(y_pred_perc, threshold, threshold)
        f1 = f1_score(y_val_arr, y_pred, average=None).tolist()
        if f1[1] >= f1_c_th:
            f1_c_th = f1[1]
            best_th_comportamenti = threshold
        if f1[3] >= f1_d_th:
            f1_d_th = f1[3]
            best_th_day = threshold
    for threshold in np.arange(0, 1, step_min_th):
        to_remove_info, _, _ = estimation_single_anomaly(y_val_arr, y_pred_perc, threshold, threshold, 0, 0, True)
        if to_remove_info[3] == 0:
            best_min_th_comportamenti = threshold
        if to_remove_info[2] == 0:
            best_min_th_day = threshold
        if to_remove_info[3] > 0 and to_remove_info[2] > 0:
            break
    return best_th_comportamenti, best_th_day, 0, 0


def threshold_medium_lvl(threshold_f1):
    return [0.05, 0.45]


def distribution_anomaly(y_model, y_target, thresholds, threshold_f1, day=False):
    soglie_errors = [[0, 0, 0, None], [None, None, 0, 0]]
    soglie_predictions = [0, 0, 0, 0]

    if day is False:
        pos_target, neg_target, name = 1, 0, 'comportamenti'
    else:
        pos_target, neg_target, name = 3, 2, 'day'

    for i in range(y_target.shape[0]):
        condition = (y_target[i] == neg_target or y_target[i] == pos_target)
        if y_model[i][pos_target] <= thresholds[0] and condition:
            soglie_predictions[0] += 1
        elif thresholds[0] < y_model[i][pos_target] <= thresholds[1] and condition:
            soglie_predictions[1] += 1
        elif thresholds[1] < y_model[i][pos_target] <= thresholds[2] and condition:
            soglie_predictions[2] += 1
        elif y_model[i][pos_target] > thresholds[2] and condition:
            soglie_predictions[3] += 1

        if y_model[i][pos_target] <= thresholds[0] and y_target[i] == pos_target:
            soglie_errors[0][0] += 1
        elif thresholds[0] < y_model[i][pos_target] <= thresholds[1] and y_target[i] == pos_target:
            soglie_errors[0][1] += 1
        elif thresholds[1] < y_model[i][pos_target] <= threshold_f1 and y_target[i] == pos_target:
            soglie_errors[0][2] += 1
        elif threshold_f1 < y_model[i][pos_target] <= thresholds[2] and y_target[i] == neg_target:
            soglie_errors[1][2] += 1
        elif y_model[i][pos_target] > thresholds[2] and y_target[i] == neg_target:
            soglie_errors[1][3] += 1
    return soglie_predictions, soglie_errors, thresholds, name


def estimation_single_anomaly(y_target, y_model, max_th_rmv_comp, max_th_rmv_day, threshold_comp, threshold_day,
                              validation=False):
    to_remove_comp, to_remove_day, to_remove_err_day, to_remove_err_comportamenti = 0, 0, 0, 0
    wrong_prediction_comp_fn, wrong_prediction_comp_fp, wrong_prediction_day_fn, wrong_prediction_day_fp = [], [], [], []
    correct_prediction_comp_tp, correct_prediction_comp_tn, correct_prediction_day_tp, correct_prediction_day_tn = [], [], [], []
    for i in range(y_target.shape[0]):
        if y_model[i][1] <= max_th_rmv_comp and y_target[i] == 0:
            to_remove_comp += 1
        elif y_model[i][3] <= max_th_rmv_day and y_target[i] == 2:
            to_remove_day += 1
        elif y_model[i][1] <= max_th_rmv_comp and y_target[i] == 1:
            to_remove_err_comportamenti += 1
        elif y_model[i][3] <= max_th_rmv_day and y_target[i] == 3:
            to_remove_err_day += 1

        if validation is True:
            continue

        if y_model[i][1] > threshold_comp and y_target[i] == 0:
            wrong_prediction_comp_fp.append(y_model[i][1])
            print(" >> false positive COMPORTAMENTI >> y_model " + str(y_model[i]) + " y_target " + str(y_target[i]))
        elif y_model[i][1] < threshold_comp and y_target[i] == 1:
            wrong_prediction_comp_fn.append(y_model[i][1])
            print(" >> false NEGATIVE COMPORTAMENTI >> y_model " + str(y_model[i]) + " y_target " + str(y_target[i]))
        elif y_model[i][3] > threshold_day and y_target[i] == 2:
            wrong_prediction_day_fp.append(y_model[i][3])
            print(" >> false positive DAY >> y_model " + str(y_model[i]) + " y_target " + str(y_target[i]))
        elif y_model[i][3] < threshold_day and y_target[i] == 3:
            wrong_prediction_day_fn.append(y_model[i][3])
            print(" >> false NEGATIVE DAY >> y_model " + str(y_model[i]) + " y_target " + str(y_target[i]))
        elif y_model[i][1] > threshold_comp and y_target[i] == 1:
            correct_prediction_comp_tp.append(y_model[i][1])
        elif y_model[i][1] < threshold_comp and y_target[i] == 0:
            correct_prediction_comp_tn.append(y_model[i][1])
        elif y_model[i][3] > threshold_day and y_target[i] == 3:
            correct_prediction_day_tp.append(y_model[i][3])
        elif y_model[i][3] < threshold_day and y_target[i] == 2:
            correct_prediction_day_tn.append(y_model[i][3])
    return [to_remove_comp, to_remove_day, to_remove_err_day, to_remove_err_comportamenti], \
           [wrong_prediction_comp_fn, wrong_prediction_comp_fp, wrong_prediction_day_fn, wrong_prediction_day_fp], \
           [correct_prediction_comp_tp, correct_prediction_comp_tn, correct_prediction_day_tp,
            correct_prediction_day_tn]


def print_single_anomaly_comparison(y_target, to_remove_info, max_th_rmv_comp, max_th_rmv_day):
    unique, counts = np.unique(y_target, return_counts=True)
    dict_unique = dict(zip(unique, counts))
    num_target_comp = dict_unique[0] + dict_unique[1]
    num_target_day = dict_unique[2] + dict_unique[3]
    print(" >> remove COMPORTAMENTI threshold " + str(max_th_rmv_comp) + " with number of elements " +
          str(to_remove_info[0]) + " >> " + str(to_remove_info[0] / num_target_comp * 100) + " %")
    print(" >> remove DAY threshold " + str(max_th_rmv_day) + " with number of elements " +
          str(to_remove_info[1]) + " >> " + str(to_remove_info[1] / num_target_day * 100) + " %")
    print(
        " >> remove COMPORTAMENTI+DAY with number of elements " + str(
            (to_remove_info[1] + to_remove_info[0])) + " >> " +
        str((to_remove_info[1] + to_remove_info[0]) / y_target.shape[0] * 100) + " %")
    print(" >> remove errors COMPORTAMENTI with number of elements " + str(to_remove_info[3]))
    print(" >> remove errors DAY with number of elements " + str(to_remove_info[2]))


def single_anomaly_comparison(y_model, y_target, threshold_comp, threshold_day, max_th_rmv_comp,
                              max_th_rmv_day, threshold_comp_med, threshold_day_med):
    np.set_printoptions(suppress=True)
    to_remove_info, wrong_prediction_info, correct_prediction_info = estimation_single_anomaly(y_target, y_model,
                                                                                               max_th_rmv_comp,
                                                                                               max_th_rmv_day,
                                                                                               threshold_comp,
                                                                                               threshold_day)

    print_single_anomaly_comparison(y_target, to_remove_info, max_th_rmv_comp, max_th_rmv_day)

    threshold_comp_list = [max_th_rmv_comp, threshold_comp_med[0], threshold_comp_med[1]]
    threshold_day_list = [max_th_rmv_day, threshold_day_med[0], threshold_day_med[1]]

    weights_comp = weights_definition(correct_prediction_info[0], correct_prediction_info[1], wrong_prediction_info[0],
                                      wrong_prediction_info[1])
    weights_day = weights_definition(correct_prediction_info[2], correct_prediction_info[3], wrong_prediction_info[2],
                                     wrong_prediction_info[3])

    plot_wrong_predictions(wrong_prediction_info[0], wrong_prediction_info[1], threshold_comp_list, threshold_comp,
                           'comportamenti', weights_comp)
    plot_wrong_predictions(wrong_prediction_info[2], wrong_prediction_info[3], threshold_day_list, threshold_day,
                           'day', weights_day)

    plot_predictions(correct_prediction_info[0], correct_prediction_info[1], wrong_prediction_info[0],
                     wrong_prediction_info[1], threshold_comp_list, threshold_comp, 'comportamenti', weights_comp)
    plot_predictions(correct_prediction_info[2], correct_prediction_info[3], wrong_prediction_info[2],
                     wrong_prediction_info[3], threshold_day_list, threshold_day, 'day', weights_day)


def model_definition_search(x_train, y_train):
    score = make_scorer(my_custom_loss_func, greater_is_better=True, needs_proba=True)
    rf_random = random_search_training(score=score)
    rf_random.fit(x_train, y_train.values.ravel())
    print(">> best parameters ", rf_random.best_params_)
    filename = 'train/models/best_params.txt'
    with open(filename, 'w') as f:
        f.write(str(rf_random.best_params_))
    return rf_random


def model_definition(x_train, y_train):
    model = compile_model_voting()
    model.fit(x_train, y_train.values.ravel())
    return model


def model_definition_nn(x_train, y_train, x_val, y_val, x_test, y_test, batch=256, epochs=100):
    label_as_binary = LabelBinarizer()

    y_train = label_as_binary.fit_transform(y_train)
    y_val = label_as_binary.fit_transform(y_val)
    y_test = label_as_binary.fit_transform(y_test)

    model, early_stopping = compile_model_nn(x_train.shape[1])

    history = model.fit(x_train, y_train, batch_size=batch, epochs=epochs, callbacks=[early_stopping],
                        validation_data=(x_val, y_val))

    plot_loss(history)
    plot_precision(history)
    plot_recall(history)

    scores = model.evaluate(x_test, y_test, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    return model


def run(nn=False):
    start = datetime.now()

    x_train_orig, y_train_orig, x_val, y_val, x_test, y_test = load_dataset_split()

    n_jobs = 1
    smote_models = [None]
    # SMOTEENN(n_jobs=n_jobs), SMOTETomek(n_jobs=n_jobs),
    # ADASYN(n_jobs=n_jobs)]

    for i in range(len(smote_models)):
        # if i > 0:
        print("\n>> smote ", str(i))
        # x_train, y_train = smote_over_sampling(x_train_orig, y_train_orig, smote_models[i])
        # else:
        x_train, y_train = x_train_orig, y_train_orig

        if nn is True:
            model = model_definition_nn(x_train, y_train, x_val, y_val, x_test, y_test)
        else:
            model = model_definition(x_train, y_train)
            # model = model_definition_search(x_train, y_train)

        threshold_comp, threshold_day, max_th_rmv_comp, max_th_rmv_day = threshold_check(x_val, y_val, model)
        threshold_comp_med, threshold_day_med = threshold_medium_lvl(threshold_comp), threshold_medium_lvl(
            threshold_day)
        all_threshold_comp = [max_th_rmv_comp, threshold_comp_med[0], threshold_comp_med[1]]
        all_threshold_day = [max_th_rmv_day, threshold_day_med[0], threshold_day_med[1]]

        print("\n>> threshold comportamenti", threshold_comp)
        print(">> threshold day", threshold_day)
        print("\n>> threshold max_th_rmv_comp", max_th_rmv_comp)
        print(">> threshold max_th_rmv_day", max_th_rmv_day)

        y_model, y_target = plots_performance(model, x_test, y_test, threshold_comp, threshold_day, all_threshold_comp,
                                              all_threshold_day)
        single_anomaly_comparison(y_model, y_target, threshold_comp, threshold_day, max_th_rmv_comp, max_th_rmv_day,
                                  threshold_comp_med, threshold_day_med)

        filename = 'train/models/voting_model.sav'
        pickle.dump(model, open(filename, 'wb'))

    print(">> total time ", datetime.now() - start, "\n")


if __name__ == "__main__":
    run()
