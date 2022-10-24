import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

plot_directory = 'C:/workspace/AnomalyPrioritization/train/plots/'
best_parameters_directory = 'C:/workspace/AnomalyPrioritization/train/model_data/best_params.txt'

name_rf, name_rf_bal, name_knn, name_dt = 'rf', 'rf_bal', 'knn', 'dt'

n_jobs = 2
verbose = 1
apply_smote = False
apply_random_search = False
show_plots = True
save_plots = False
x_size_plot = 18
y_size_plot = 10

# smote oversampling (SMOTETomek, ADASYN, SMOTEENN)

smote_model = SMOTEENN(n_jobs=n_jobs)

# classifiers

rf_default = RandomForestClassifier()

dt_default = DecisionTreeClassifier()

rf_bal_default = RandomForestClassifier(class_weight='balanced')

rf_bal = RandomForestClassifier(n_estimators=200, class_weight='balanced')

rf = RandomForestClassifier(n_estimators=200)

dt = DecisionTreeClassifier(class_weight='balanced')

knn = KNeighborsClassifier()

svc = SVC(kernel='rbf', probability=True, class_weight='balanced')

gbc = GradientBoostingClassifier(n_estimators=150, verbose=1)

# voting classifier

voting_estimators = [(name_rf, rf), (name_rf_bal, rf_bal), (name_knn, knn), (name_dt, dt)]
weights = [8, 6, 2, 3.5]
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

path_save_errors = "C:/workspace/AnomalyPrioritization/test/errors.csv"
soglie = ['Irrilevante', 'Bassa', 'Media', 'Alta']
col_names = ['ID', 'Predizione percentuale [%]', 'Predizione', 'Priorità', 'Effettivo', 'Software']
da_segnalare_name, da_non_segnalare_name = "Da segnalare", "Non da segnalare"
