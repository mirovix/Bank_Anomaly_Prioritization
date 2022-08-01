#!/usr/bin/env python3

"""
@Author: Miro
@Date: 17/06/2022
@Version: 1.1
@Objective: confusion matrix, roc curve, and feature importance plots
@TODO:
"""

import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay, precision_recall_fscore_support, auc
import seaborn as sns


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion_matrix',
                          c_map=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=c_map)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_roc(y_pred, y_test_arr, pos_label, num=3):
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)

    y_pred = np.array_split(y_pred, num)
    y_test_arr = np.array_split(y_test_arr, num)

    fig, ax = plt.subplots(figsize=(15, 8))
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
        print(">> evaluation per fold label "+str(pos_label)+" "+str(precision_recall_fscore_support(y_test_arr[i], y_pred[i])))
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
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

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="",
    )
    ax.legend(loc="lower right")
    plt.show()


def plot_feature_importance(importance, names, model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    plt.figure(figsize=(50, 80))

    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()
