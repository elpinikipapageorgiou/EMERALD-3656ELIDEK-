# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:54:16 2023

@author: japostol
"""

import sys
sys.path.append('C:\\Users\\japostol\\Desktop\\HumanAI codes\\')
import pandas as pd

import scikitplot as skplt
import spn_clinical_models
import spn_model_evaluation_plots
import matplotlib.pyplot as plt
from copy import deepcopy

import spn_clinical_functions

# Load Data
csv_file_path = 'C:\\Users\\japostol\\Desktop\\HumanAI codes\\labels.xlsx'  # Replace 'your_file.csv' with the actual file path
data = pd.read_excel(csv_file_path)
data = data.sample(frac=1).reset_index(drop=True)

original_data = deepcopy(data)
# Delete the ID, not needed
data=data.drop('ID',axis=1)
data=data.drop('AGE',axis=1)
data=data.drop('Gender',axis=1)
data=data.drop('SUV',axis=1)
data=data.drop('GLU',axis=1)
data=data.drop('BMI',axis=1)
data=data.drop('LABEL BASED ON BIOPSY (1), FOLLOW-UP (2), DOCTOR (3)',axis=1)
# data=data.drop('Doctor 1',axis=1)
# data=data.drop('Doctor 2',axis=1)




# FEATURES

selected_features = []
#%%
# Assuming the last column is the target variable and the rest are features
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable
test = X

AVAILABLE_CLASSIFIERS = ['catboost','logistic','bayes','knn','rf','xgb','lightgbm','svm','nn','adaboost','lda']
# excluded for now NN
classifier_name = 'xgb'

#%% GRID SEARCH

# grid_search_model = spn_clinical_functions.grid_search(classifier_name,X,y,test,selected_features)


#%%
# 10F
all_predictions,all_true_labels,classifier,Xen,yen,all_predictions_proba = spn_clinical_functions.train_kfold (classifier_name,X,y,selected_features)


import numpy as np
all_predictions_proba = np.array(all_predictions_proba)
#kfold_metrics = spn_clinical_functions.metrics(all_predictions,all_true_labels,all_predictions_proba[:,1])

kfold_metrics, metrics_df, metric_names, metric_values = spn_clinical_functions.calculate_metrics(y_true=all_true_labels, y_pred=all_predictions_proba[:,1], 
                                                                          y_pred_binary=all_predictions, positive_class=1)



spn_clinical_functions.print_metrics (kfold_metrics)

#%%

# PLOT THE LEARNING CURVE
classifier_untrained = spn_clinical_models.selector(classifier_name='xgb')
spn_model_evaluation_plots.plot_learning_curve_scikit(classifier_untrained, Xen, yen)


# PLOT THE ROC

spn_model_evaluation_plots.plot_roc_scikit(all_predictions_proba,all_true_labels)


# PLOT KS
spn_model_evaluation_plots.plot_ks_statistic(all_true_labels,all_predictions_proba)


# RELIABILITY CURVE FOR ONE
clf_names = ['XGBoost']
prediction_list = [all_predictions_proba]
trues = all_true_labels
spn_model_evaluation_plots.plot_reliability_curve(prediction_list,trues,clf_names)

kappa_score, observed_agreement, expected_agreement = spn_model_evaluation_plots.calculate_cohens_kappa_matrices(all_true_labels, all_predictions)



#%%
# FIT
all_predictions,all_true_labels,classifier,importance,Xen,y_new,all_predictions_proba = spn_clinical_functions.fit(classifier_name,X,y,test,selected_features)
import numpy as np
all_predictions_proba = np.array(all_predictions_proba)
# fit_metrics = spn_clinical_functions.metrics(all_predictions,all_true_labels,all_predictions_proba[:,1])
fit_metrics, metrics_df, metric_names, metric_values = spn_clinical_functions.calculate_metrics(y_true=all_true_labels, y_pred=all_predictions_proba[:,1], 
                                                                          y_pred_binary=all_predictions, positive_class=1)


spn_clinical_functions.print_metrics (fit_metrics)

if classifier_name == 'xgb':
    importance = classifier.feature_importances_
    features = Xen.columns
    IMPORTANCES = pd.DataFrame(importance, index=features)


# PLOT THE FEATURE IMPORRTANCE
trained_classifier = classifier
spn_model_evaluation_plots.plot_feature_importance(trained_classifier,feature_names=selected_features)


# PLOT THE ROC

spn_model_evaluation_plots.plot_roc_scikit(all_predictions_proba,all_true_labels)


# PLOT KS
spn_model_evaluation_plots.plot_ks_statistic(all_true_labels,all_predictions_proba)


# RELIABILITY CURVE FOR ONE
clf_names = ['XGBoost']
prediction_list = [all_predictions_proba]
trues = all_true_labels
spn_model_evaluation_plots.plot_reliability_curve(prediction_list,trues,clf_names)


# COHENS
kappa_score, observed_agreement, expected_agreement = spn_model_evaluation_plots.calculate_cohens_kappa_matrices(all_true_labels, all_predictions)
