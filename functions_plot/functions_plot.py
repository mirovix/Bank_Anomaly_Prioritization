#!/usr/bin/env python3

"""
@Author: Miro
@Date: 17/06/2022
@Version: 1.2
@Objective: confusion matrix, roc curve, and feature importance plots
@TODO:
"""

import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay, precision_recall_fscore_support, auc
import seaborn as sns
from configs import train_config as tc


def plot_confusion_matrix(cm, classes, normalize=False, title='confusion_matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\n >> normalized " + title)
    else:
        print("\n>> " + title)

    print(str(cm))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    value_format = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], value_format),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > (cm.max() / 2.) else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if tc.save_plots is True:
        plt.savefig(tc.plot_directory + title + ".png")
    if tc.show_plots is True:
        plt.show()


def plot_roc(y_pred, y_test_arr, pos_label, num=3):
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)

    y_pred = np.array_split(y_pred, num)
    y_test_arr = np.array_split(y_test_arr, num)

    fig, ax = plt.subplots(figsize=(tc.x_size_plot, tc.y_size_plot))
    for i in range(num):
        viz = RocCurveDisplay.from_predictions(
            y_true=y_test_arr[i],
            y_pred=y_pred[i],
            name="ROC fold {}".format(i),
            alpha=0.5,
            lw=1,
            pos_label=pos_label,
            ax=ax,
        )
        print(">> evaluation per fold label " + str(pos_label) + " " + str(
            precision_recall_fscore_support(y_test_arr[i], y_pred[i])))
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = float(np.std(aucs))
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Avg. ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="",)
    ax.legend(loc="lower right")
    if tc.save_plots is True:
        plt.savefig(tc.plot_directory + "roc_" + str(pos_label) + ".png")
    if tc.show_plots is True:
        plt.show()


def plot_feature_importance(importance, names):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    plt.figure(figsize=(50, 80))

    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    if tc.show_plots is True:
        plt.show()


def weights_definition(correct_prediction_tp, correct_prediction_tn, wrong_prediction_fn, wrong_prediction_fp):
    len_predictions = (len(correct_prediction_tp + correct_prediction_tn + wrong_prediction_fn + wrong_prediction_fp))
    weights = []
    for prediction in [correct_prediction_tp, correct_prediction_tn, wrong_prediction_fn, wrong_prediction_fp]:
        weights.append(np.ones_like(prediction) / len_predictions)
    return weights


def threshold_legend_plot(ax, threshold, f1_threshold, legend_1=None, legend_2=None):
    if legend_2 is None: legend_2 = []
    if legend_1 is None: legend_1 = []

    for i, item in enumerate(threshold):
        ax[0].axvline(item, color='black')
        legend_1.append('soglia ' + str(i) + ' ' + str("{:10.2f}".format(item)))
        ax[1].axvline(item, color='black')
        legend_2.append('soglia ' + str(i) + ' ' + str("{:10.2f}".format(item)))

    ax[0].axvline(f1_threshold, color='blue')
    legend_1.append('soglia f1 ' + str("{:10.2f}".format(f1_threshold)))
    ax[1].axvline(f1_threshold, color='blue')
    legend_2.append('soglia f1 ' + str("{:10.2f}".format(f1_threshold)))

    return ax, legend_1, legend_2


def plot_wrong_predictions(input_plot, bins=20):
    wrong_prediction_fn, wrong_prediction_fp, threshold, f1_threshold, name, weights = input_plot

    fig, ax = plt.subplots(2, 1)
    hist_fn, _, patches_fn = ax[0].hist([wrong_prediction_fn], bins, weights=[weights[2]])
    hist_fp, _, patches_fp = ax[1].hist([wrong_prediction_fp], bins, weights=[weights[3]])

    ax[0].cla()
    ax[1].cla()

    ax[0].hist([wrong_prediction_fn], bins, alpha=0.75, label=['fn'], color=['red'])
    ax[1].hist([wrong_prediction_fp], bins, alpha=0.75, label=['fp'], color=['red'])

    legend_fn = ['fn ' + str("{:10.3f}".format(np.sum(hist_fn) * 100)) + '%']
    legend_fp = ['fp ' + str("{:10.3f}".format(np.sum(hist_fp) * 100)) + '%']
    ax, legend_fn, legend_fp = threshold_legend_plot(ax, threshold, f1_threshold, legend_fn, legend_fp)

    for e in ax:
        e.set_xlabel('Distribuzione delle probabilità')
        e.set_ylabel('Numero elementi')
        e.set_xlim(-0.05, 1)

    ax[0].set_title('Istogramma dei falsi negativi discovery ' + name)
    ax[0].legend(legend_fn, loc='upper left')
    ax[1].set_title('Istogramma dei falsi positivi discovery ' + name)
    ax[1].legend(legend_fp, loc='upper right')

    plt.subplots_adjust(hspace=0.5)
    fig.set_size_inches(tc.x_size_plot, tc.y_size_plot)
    if tc.save_plots is True:
        plt.savefig(tc.plot_directory + "wrong_prediction_" + name + ".png")
    if tc.show_plots is True:
        plt.show()


def plot_predictions(input_plot, bins=20):
    correct_prediction_tp, correct_prediction_tn, threshold, f1_threshold, name, weights = input_plot

    fig, ax = plt.subplots(2, 1)
    ax[0].hist([correct_prediction_tp], bins, alpha=0.75, label=['tp'], color=['green'], weights=[weights[0]])
    ax[1].hist([correct_prediction_tn], bins, alpha=0.75, label=['tn'], color=['green'], weights=[weights[1]])

    legend_tp, legend_tn = ['tp'], ['tn']
    ax, legend_tp, legend_tn = threshold_legend_plot(ax, threshold, f1_threshold, legend_tp, legend_tn)
    ax[0].legend(legend_tp, loc='upper left')
    ax[1].legend(legend_tn, loc='upper right')

    for e in ax:
        e.set_xlabel('Distribuzione delle probabilità')
        e.set_ylabel('Probabilità')
        e.set_xlim(-0.05, 1)

    ax[0].set_title('Istogramma veri positivi del discovery ' + name)
    ax[1].set_title('Istogramma veri negativi del discovery ' + name)

    plt.subplots_adjust(hspace=0.8)
    fig.set_size_inches(16, 8)
    if tc.save_plots is True:
        plt.savefig(tc.plot_directory + "correct_prediction_" + name + ".png")
    if tc.show_plots is True:
        plt.show()

    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(
        [correct_prediction_tp, correct_prediction_tn], bins, alpha=0.75,
        label=['tn', 'tp', 'fn', 'fp'], color=['green', 'green'], weights=[weights[0], weights[1]])

    for p in patches:
        p.datavalues *= 100
        ax.bar_label(p, fmt='%.3f')

    fig.set_size_inches(tc.x_size_plot, tc.y_size_plot)
    plt.axvline(f1_threshold, label='threshold')
    ax.set_xlabel('Bins')
    ax.set_ylabel('Probability')
    ax.set_title('Histogram of predictions ' + name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    if tc.save_plots is True:
        plt.savefig(tc.plot_directory + "all_correct_prediction_" + name + ".png")
    if tc.show_plots is True:
        plt.show()


def plot_table_soglie(input_data):
    soglie_predictions, soglie_errors, _, name = input_data
    fig, ax = plt.subplots(1, 1)
    data = [[soglie_predictions[0], soglie_errors[0][0], soglie_errors[1][0], 'Molto Bassa'],
            [soglie_predictions[1], soglie_errors[0][1], soglie_errors[1][1], 'Bassa'],
            [soglie_predictions[2], soglie_errors[0][2], soglie_errors[1][2], 'Media'],
            [soglie_predictions[3], soglie_errors[0][3], soglie_errors[1][3], 'Alta']]
    column_labels = ["Predizioni", "Errori (fn)", "Errori (fp)", "Priorità"]
    ax.set_title('Tabella soglie del discovery ' + name)
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=data, colLabels=column_labels, loc="center")
    if tc.save_plots is True:
        plt.savefig(tc.plot_directory + "table_" + name + ".png")
    if tc.show_plots is True:
        plt.show()


def plot_loss(history):
    plt.figure(figsize=(tc.x_size_plot, tc.y_size_plot))
    plt.plot(history.epoch, history.history['loss'], label='loss')
    plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
    plt.title('loss')
    plt.legend()
    if tc.show_plots is True:
        plt.show()


def plot_precision(history):
    plt.figure(figsize=(tc.x_size_plot, tc.y_size_plot))
    plt.plot(history.epoch, history.history['precision_1'], label='precision_1')
    plt.plot(history.epoch, history.history['val_precision_1'], label='val_precision_1')
    plt.plot(history.epoch, history.history['precision_3'], label='precision_3')
    plt.plot(history.epoch, history.history['val_precision_3'], label='val_precision_3')
    plt.title('precision')
    plt.legend()
    if tc.show_plots is True:
        plt.show()


def plot_recall(history):
    plt.figure(figsize=(tc.x_size_plot, tc.y_size_plot))
    plt.plot(history.epoch, history.history['recall_1'], label='recall_1')
    plt.plot(history.epoch, history.history['val_recall_1'], label='val_recall_1')
    plt.plot(history.epoch, history.history['recall_3'], label='recall_3')
    plt.plot(history.epoch, history.history['val_recall_3'], label='val_recall_3')
    plt.title('recall')
    plt.legend()
    if tc.show_plots is True:
        plt.show()
