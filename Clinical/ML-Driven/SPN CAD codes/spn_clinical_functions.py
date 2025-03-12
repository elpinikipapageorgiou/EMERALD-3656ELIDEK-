# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:38:33 2023

@author: japostol
"""

import sys
sys.path.append('C:\\Users\\japostol\\Desktop\\SPN Clinical Python codes\\')
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import spn_clinical_models
import numpy as np
import pandas as pd
import sklearn
#%%

def select_features(data,selected_features):
    if selected_features == '' or selected_features==[]:
        return data
    new_data = data[selected_features]
    return new_data

def fit(classifier_name,X,y,test,selected_features):
    
    classifiers_needing_label_encoding = ['catboost','logistic','bayes','knn','rf','xgb','lightgbm','svm','nn','adaboost','gmm','lda','elastic']  
    if classifier_name in classifiers_needing_label_encoding:    
        # Label encode the target variable y if it contains non-numerical values
        label_encoder = LabelEncoder()
        yen = label_encoder.fit_transform(y)
        
        # Convert non-numerical columns in X to numerical using one-hot encoding
        Xen = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)
        test = pd.get_dummies(test, columns=test.select_dtypes(include=['object']).columns)
    else:
        Xen = X
        yen = y
        
    Xen = select_features(Xen,selected_features)
    test = select_features(test,selected_features)
    classifier = spn_clinical_models.selector(classifier_name)
    classifier.fit(Xen, yen)
    #importance = classifier.feature_importances_
    importance = None
    ypred = classifier.predict(test)
    ypred_proba = classifier.predict_proba(test)
    
    return ypred,yen,classifier,importance, Xen, y,ypred_proba

def grid_search(classifier_name,X,y,test,selected_features):
    
    from sklearn.model_selection import GridSearchCV
    
    classifiers_needing_label_encoding = ['catboost','logistic','bayes','knn','rf','xgb','lightgbm','svm','nn','adaboost','gmm','lda','elastic']  
    if classifier_name in classifiers_needing_label_encoding:    
        # Label encode the target variable y if it contains non-numerical values
        label_encoder = LabelEncoder()
        yen = label_encoder.fit_transform(y)
        
        # Convert non-numerical columns in X to numerical using one-hot encoding
        Xen = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)
        test = pd.get_dummies(test, columns=test.select_dtypes(include=['object']).columns)
    else:
        Xen = X
        yen = y
    Xen = select_features(Xen,selected_features)
    test = select_features(test,selected_features)
    classifier = spn_clinical_models.selector(classifier_name)
    
    
    if classifier_name == 'rf':
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 300, 600],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Initialize GridSearchCV
        grid_search_model = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='recall')
        
        # Perform grid search
        grid_search_model.fit(Xen, yen)
        
        # Get the best parameters
        best_params = grid_search_model.best_params_
        print("Best Parameters for {}:".format(classifier_name), best_params)
    
    if classifier_name == 'xgb':
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [200, 400, 600],
            'learning_rate': [0.01, 0.1],
            'max_depth': [ 4, 5, 7, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 1.0]
        }
        
        # Initialize GridSearchCV
        grid_search_model = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=10, scoring='recall') # roc_auc or accuracy or recall in scoring
        
        # Perform grid search
        grid_search_model.fit(Xen, yen)
        
        # Get the best parameters
        best_params = grid_search_model.best_params_
        print("Best Parameters for {}:".format(classifier_name), best_params)   
    
    
    if classifier_name == 'catboost':
        # Define hyperparameter grid
        param_grid = {
            'iterations': [50, 100, 200],
            'depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5]
        }
        
        # Initialize GridSearchCV
        grid_search_model = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='recall')
        
        # Perform grid search
        grid_search_model.fit(Xen, yen)
        
        # Get the best parameters
        best_params = grid_search_model.best_params_
        print("Best Parameters for {}:".format(classifier_name), best_params)
    
        
    if classifier_name == 'logistic':
        # Define hyperparameter grid
        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga']
        }
        
        # Initialize GridSearchCV
        grid_search_model = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='recall')
        
        # Perform grid search
        grid_search_model.fit(Xen, yen)
        
        # Get the best parameters
        best_params = grid_search_model.best_params_
        print("Best Parameters for {}:".format(classifier_name), best_params)       

        
    if classifier_name == 'knn':
        # Define hyperparameter grid
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
        
        # Initialize GridSearchCV
        grid_search_model = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='recall')
        
        # Perform grid search
        grid_search_model.fit(Xen, yen)
        
        # Get the best parameters
        best_params = grid_search_model.best_params_
        print("Best Parameters for {}:".format(classifier_name), best_params)   

    if classifier_name == 'lightgbm':
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [ 100, 200, 400, 600],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [5, 10, 15],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
        
        # Initialize GridSearchCV
        grid_search_model = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='recall')
        
        # Perform grid search
        grid_search_model.fit(Xen, yen)
        
        # Get the best parameters
        best_params = grid_search_model.best_params_
        print("Best Parameters for {}:".format(classifier_name), best_params)

    if classifier_name == 'svm':
        # Define hyperparameter grid
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto']
        }
        
        # Initialize GridSearchCV
        grid_search_model = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='recall')
        
        # Perform grid search
        grid_search_model.fit(Xen, yen)
        
        # Get the best parameters
        best_params = grid_search_model.best_params_
        print("Best Parameters for {}:".format(classifier_name), best_params)

    if classifier_name == 'adaboost':
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        
        # Initialize GridSearchCV
        grid_search_model = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='recall')
        
        # Perform grid search
        grid_search_model.fit(Xen, yen)
        
        # Get the best parameters
        best_params = grid_search_model.best_params_
        print("Best Parameters for {}:".format(classifier_name), best_params)

    if classifier_name == 'gmm':
        # Define hyperparameter grid
        param_grid = {
            'n_components': [2, 3, 4, 5],
            'covariance_type': ['full', 'tied', 'diag', 'spherical']
        }
        
        # Initialize GridSearchCV
        grid_search_model = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='recall')
        
        # Perform grid search
        grid_search_model.fit(Xen, yen)
        
        # Get the best parameters
        best_params = grid_search_model.best_params_
        print("Best Parameters for {}:".format(classifier_name), best_params)

    if classifier_name == 'elastic':
        # Define hyperparameter grid
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1, 10],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        
        # Initialize GridSearchCV
        grid_search_model = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='recall')
        
        # Perform grid search
        grid_search_model.fit(Xen, yen)
        
        # Get the best parameters
        best_params = grid_search_model.best_params_
        print("Best Parameters for {}:".format(classifier_name), best_params)
    
    return grid_search_model

def train_kfold (classifier_name,X,y,selected_features):
    
    classifiers_needing_label_encoding = ['catboost','logistic','bayes','knn','rf','xgb','lightgbm','svm','nn','adaboost','gmm','lda','elastic'] 
    if classifier_name in classifiers_needing_label_encoding:
        # Label encode the target variable y if it contains non-numerical values
        label_encoder = LabelEncoder()
        yen = label_encoder.fit_transform(y)
        
        # Convert non-numerical columns in X to numerical using one-hot encoding
        Xen = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)
    else:
        Xen = X
        yen = y
    
    Xen = select_features(Xen,selected_features)
    
    kf = KFold(n_splits=10, shuffle=False)
    all_predictions = []
    all_predictions_proba = []
    all_true_labels = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(Xen)):
        X_train, X_test = Xen.iloc[train_idx], Xen.iloc[test_idx]
        y_train, y_test = yen[train_idx], yen[test_idx]
    
        # Fit the model and obtain predictions for the fold
        classifier = spn_clinical_models.selector(classifier_name)
        classifier.fit(X_train, y_train)
        fold_predictions = classifier.predict(X_test)
        fold_predictions_proba = classifier.predict_proba(X_test)
        
        
        # calibrate
        # SOS! This replaces the probabilities with new probabilites, but retains the binary output of the original classifier
        calibrated_classifier = CalibratedClassifierCV(classifier, method='sigmoid', cv='prefit')
        calibrated_classifier.fit(X_train, y_train)

        # Predict probabilities on the test set
        fold_predictions_proba = calibrated_classifier.predict_proba(X_test)
        
        # Store predictions and true labels for later evaluation
        all_predictions.extend(fold_predictions)
        all_predictions_proba.extend(fold_predictions_proba)
        all_true_labels.extend(y_test)
    
    
    return all_predictions,all_true_labels,classifier,Xen,yen,all_predictions_proba
    
def metrics(all_predictions,all_true_labels,all_predictions_proba):  
    
    '''
    
    THESE ARE THE OLD METRICS - NOTHING BAD, JUST FEWERE METRICS
    
    '''
    
    
    # Function to calculate accuracy
    def calculate_accuracy(all_predictions, all_true_labels):
        return accuracy_score(all_true_labels, all_predictions)
    
    # Function to calculate sensitivity (recall for class 1)
    def calculate_sensitivity(all_predictions, all_true_labels):
        return recall_score(all_true_labels, all_predictions, pos_label=1)
    
    # Function to calculate specificity (recall for class 0)
    def calculate_specificity(all_predictions, all_true_labels):
        tn, fp, fn, tp = confusion_matrix(all_true_labels, all_predictions).ravel()
        return tn / (tn + fp), tp,tn,fp,fn
    
    # Function to calculate recall (class 1)
    def calculate_recall(all_predictions, all_true_labels):
        return recall_score(all_true_labels, all_predictions)
    
    # Function to calculate precision (positive predictive value)
    def calculate_precision(all_predictions, all_true_labels):
        return precision_score(all_true_labels, all_predictions)
    
    # Function to calculate AUC score
    def calculate_auc_score(all_predictions, all_true_labels):
        return roc_auc_score(all_true_labels, all_predictions)
    
    # Example usage:
    accuracy = calculate_accuracy(all_predictions, all_true_labels)
    sensitivity = calculate_sensitivity(all_predictions, all_true_labels)
    specificity,tp,tn,fp,fn = calculate_specificity(all_predictions, all_true_labels)
    recall = calculate_recall(all_predictions, all_true_labels)
    precision = calculate_precision(all_predictions, all_true_labels)
    auc_score = calculate_auc_score(all_predictions_proba, all_true_labels)
    
    # Display results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    
    
    metrics = {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Recall': recall,
        'Precision': precision,
        'AUC': auc_score,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        }
    
    return metrics


def calculate_metrics(y_true, y_pred, y_pred_binary, positive_class=1):
    
    # Convert labels to arrays if they are not
    y_true = np.array(y_true)
    y_pred_numeric = np.array(y_pred)  # Numeric predictions
    #y_pred_binary = np.where(y_pred_numeric >= 0.5, 1, 0)  # Binary predictions based on threshold 0.5

    # Ensure labels are in 1D format
    if len(y_true.shape) > 1 and y_true.shape[1] == 1:
        y_true = y_true.flatten()

    # Convert one-hot encoded labels to binary
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)

    # Confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred_binary)

    # True Positives, True Negatives, False Positives, False Negatives
    tp = confusion_matrix[positive_class, positive_class]
    tn = np.sum(confusion_matrix) - np.sum(confusion_matrix[positive_class, :]) - np.sum(confusion_matrix[:, positive_class]) + tp
    fp = np.sum(confusion_matrix[:, positive_class]) - tp
    fn = np.sum(confusion_matrix[positive_class, :]) - tp

    # Classification Metrics (using binary predictions)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred_binary)
    precision = sklearn.metrics.precision_score(y_true, y_pred_binary, pos_label=positive_class)
    recall = sklearn.metrics.recall_score(y_true, y_pred_binary, pos_label=positive_class)
    f1 = sklearn.metrics.f1_score(y_true, y_pred_binary, pos_label=positive_class)

    # Sensitivity (True Positive Rate or Recall)
    sensitivity = recall

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp)

    # Positive Predictive Value (Precision)
    ppv = precision

    # Negative Predictive Value
    npv = tn / (tn + fn)

    # True Positive Rate (Sensitivity or Recall)
    tpr = sensitivity

    # True Negative Rate (Specificity)
    tnr = specificity
    
    # False Positive Rate
    fpr = fp / (fp + tn)
    
    # False Negative Rate\
    fnr = fn / (fn + tp)

    # AUC Score (using numeric predictions)
    auc = sklearn.metrics.roc_auc_score(y_true, y_pred_numeric)

    # Cohen's Kappa Score
    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred_binary)

    # Dictionary of Metrics
    metrics_dict = {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC Score': auc,
        'Cohen\'s Kappa': kappa,
        'True Positives': tp,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'True Positive Rate': tpr,
        'True Negative Rate': tnr,
        'False Positive Rate': fpr,
        'False Negative Rate': fnr,
        'Positive Predictive Value': ppv,
        'Negative Predictive Value': npv,
        #'Confusion Matrix': confusion_matrix
    }

    # Convert to Pandas DataFrame
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'Value'])

    # Separate lists of names and values
    metric_names = metrics_df['Metric'].tolist()
    metric_values = metrics_df['Value'].tolist()

    return metrics_dict, metrics_df, metric_names, metric_values





    
    
def print_metrics (metrics):
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print ('')
    print ('')
    for key, value in metrics.items():
        print(f"{value}")   
    
    
    
    
    
    
    
    
    
    
    
    
    
    