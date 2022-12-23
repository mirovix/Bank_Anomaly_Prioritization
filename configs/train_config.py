"""
@Author: Miro
@Date: 26/10/2022
@Version: 1.0
@Objective: configuration file for training
@TODO:
"""

import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from configs import production_config as pc

plot_directory = pc.base_path + '/plots/'
best_parameters_directory = pc.base_path + '/model_data/best_params.txt'

name_rf, name_rf_bal, name_knn, name_dt, name_sgd = 'rf', 'rf_bal', 'knn', 'dt', 'sgd'

n_jobs = 3
verbose = 1
apply_smote = False
apply_random_search = False
show_plots = True
save_plots = True
x_size_plot = 18
y_size_plot = 10

fixed_threshold_mid = [0.12, 0.52]

# smote oversampling (SMOTETomek, ADASYN, SMOTEENN, SVMSMOTE, BorderlineSMOTE)

smote_model = BorderlineSMOTE(n_jobs=n_jobs, random_state=1)

# classifiers

rf_default = RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=n_jobs)

dt_default = DecisionTreeClassifier(random_state=1)

rf_bal_default = RandomForestClassifier(class_weight='balanced', random_state=1, n_jobs=n_jobs)

rf_bal = RandomForestClassifier(n_estimators=450, class_weight='balanced', random_state=1, bootstrap=True,
                                min_samples_split=25, min_samples_leaf=25, n_jobs=n_jobs)

rf = RandomForestClassifier(n_estimators=450, random_state=1, bootstrap=True, n_jobs=n_jobs)

dt = DecisionTreeClassifier(class_weight='balanced', random_state=1)

knn = KNeighborsClassifier(n_neighbors=8, n_jobs=n_jobs, weights='distance')

svc = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=1)

gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, verbose=1, random_state=1)

hgbc = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, verbose=1, random_state=1,
                                      l2_regularization=0.02, validation_fraction=0.15,
                                      min_samples_leaf=20, max_leaf_nodes=80, n_iter_no_change=15)

qda = QuadraticDiscriminantAnalysis()

abc = AdaBoostClassifier()

sgd = SGDClassifier(loss='log_loss')

et = ExtraTreesClassifier(200)

# voting classifier

voting_estimators = [(name_rf, rf), (name_rf_bal, rf_bal), (name_knn, knn), (name_dt, dt), (name_sgd, gbc)]
weights = [9, 4, 2, 2.5, 4]
voting = 'soft'

# random search parameters

score_rs = 'f1'
n_iter_rs = 1000
cv_rs = 5
n_estimators_rs = [int(x) for x in np.linspace(start=100, stop=1500, num=10)]
class_weight_rs = ['balanced', None]
max_depth_rs = [int(x) for x in np.linspace(40, 140, num=8)]
max_depth_rs.append(None)
min_samples_rs = [2, 3, 5, 0.01, 0.1, 0.2]
criterion_rs = ["gini", "entropy", "log_loss"]
max_features_rs = ["sqrt", "log2"]
weights_rs = []
for i in range(2, 13, 2):
    for j in range(1, 5):
        for k in range(1, 7, 2):
            weights_rs.append([i, j, k])
voting_estimators_rs = [(name_rf, rf_default), (name_rf_bal, rf_bal_default), (name_dt, dt_default)]

random_grid = {
    'weights': weights_rs,
    name_rf + '__n_estimators': n_estimators_rs,
    # name_rf+'__class_weight': class_weight_rs,
    name_rf + '__max_depth': max_depth_rs,
    # name_rf+'__min_samples_split': train_config.min_samples_rs,
    # name_rf+'__min_samples_leaf': train_config.min_samples_rs,
    name_rf + '__max_features': max_features_rs,
    name_rf + '__criterion': criterion_rs,
    name_dt + '__criterion': criterion_rs,
    # name_dt+'__splitter': splitter_rs,
    name_dt + '__max_depth': max_depth_rs,
    # name_dt+'__min_samples_leaf': min_samples_rs,
    # name_dt+'__min_samples_split': min_samples_rs,
    name_dt + '__class_weight': class_weight_rs,
    name_dt + '__max_features': max_features_rs,
    name_rf_bal + '__max_depth': max_depth_rs,
    name_rf_bal + '__min_samples_split': min_samples_rs,
    # name_rf_bal+'__min_samples_leaf': min_samples_rs,
    name_rf_bal + '__max_features': max_features_rs,
    name_rf_bal + '__criterion': criterion_rs,
}

# error performance data configuration

path_save_errors = pc.base_path + "/test/errors.csv"
soglie = ['Irrilevante', 'Bassa', 'Media', 'Alta']
production_soglie = {0: 4, 1: 3, 2: 2, 3: 1, pc.not_predicted_value_fascia: 9}
col_names = ['ID', 'Predizione percentuale [%]', 'Predizione', 'Priorità', 'Effettivo', 'Software']
da_segnalare_name, da_non_segnalare_name = "Da segnalare", "Non da segnalare"
