import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
import xgboost
from sklearn import tree
import shap
from sklearn.tree import DecisionTreeClassifier
import sys
import seaborn as sns
import joblib

# function for printing each component of confusion matrix
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

# function to print SHAP values and plots for tree based algorithms
def xai(model, X, idx):
    explainer = shap.KernelExplainer(model.predict, X.values[idx])
    shap_values = explainer.shap_values(X)
    print(shap_values.shape)
    # shap.summary_plot(shap_values, X)
    # shap.summary_plot(shap_values, X, plot_type='violin')
    # for feature in X.columns:
    #     print(feature)
    #     shap.dependence_plot(feature, shap_values, X)
    # idx_ben = 4 # datapoint to explain (benign)
    # idx_mal = 1 # datapoint to explain (malignant)
    # sv = explainer.shap_values(X.loc[[idx_ben]])
    # exp = shap.Explanation(sv,explainer.expected_value, data=X.loc[[idx_ben]].values, feature_names=X.columns)
    # shap.waterfall_plot(exp[0])
    # sv = explainer.shap_values(X.loc[[idx_mal]])
    # exp = shap.Explanation(sv,explainer.expected_value, data=X.loc[[idx_mal]].values, feature_names=X.columns)
    # shap.waterfall_plot(exp[0])
    # shap.force_plot(explainer.expected_value, shap_values[idx_ben,:], X.iloc[idx_ben,:], matplotlib=True)
    # shap.force_plot(explainer.expected_value, shap_values[idx_mal,:], X.iloc[idx_mal,:], matplotlib=True)
    ###
    # shap.decision_plot(0, shap_values, X.loc[idx])
    # shap.decision_plot(0, shap_values[idx_ben], X.loc[idx[idx_ben]], highlight=0)
    # shap.decision_plot(0, shap_values[idx_mal], X.loc[idx[idx_mal]], highlight=0)
    return shap_values

def xai_cat(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    sv = explainer.shap_values(X.loc[[0]])
    exp = shap.Explanation(sv,explainer.expected_value, data=X.loc[[0]].values, feature_names=X.columns)
    # Show waterfall plot for idx_cad
    # fig_cad, _ = plt.subplots()
    shap.waterfall_plot(exp[0])

data_path = 'nsclc_loc.csv'
data = pd.read_csv(data_path, na_filter = False)
dataframe = pd.DataFrame(data.values, columns=data.columns)
x = dataframe.drop(['Output'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
y = dataframe['Output']

# ml algorithms initialization
svmc = svm.SVC(kernel='rbf') # 77.87%, STD 11.373
dt = DecisionTreeClassifier() # 91.47%, STD 7.523
rndF = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=36) # 93.1%, STD 6.231 [*]
ada = AdaBoostClassifier(n_estimators=30, random_state=0) # 94.33%, STD 6.3 [*]
knn = KNeighborsClassifier(n_neighbors=7) # 73.27%, STD 7.177
tab = TabPFNClassifier(device='cpu', N_ensemble_configurations=8) # 91.87%, STD 7.414 [*]
xgb = xgboost.XGBRegressor(objective="binary:hinge", random_state=42) # 88.93%, STD 5.446
light = LGBMClassifier(objective='binary', random_state=5, n_estimators=25, n_jobs=-1) # 92,27 6,365 [*]
catb = CatBoostClassifier(n_estimators=79, learning_rate=0.1, verbose=False) # 92.25%, STD 6.42

sel_alg = catb
X = x

est = sel_alg.fit(X, y)
n_yhat = cross_val_predict(sel_alg, X, y, cv=10)

print("cv-10 accuracy: ", cross_val_score(sel_alg, X, y, scoring='accuracy', cv = 10).mean() * 100)
print("cv-10 accuracy STD: ", cross_val_score(sel_alg, X, y, scoring='accuracy', cv = 10).std() * 100)
scoring = {
    'sensitivity': metrics.make_scorer(metrics.recall_score),
    'specificity': metrics.make_scorer(metrics.recall_score,pos_label=0)
}
print("sensitivity: ", cross_val_score(sel_alg, X, y, scoring=scoring['sensitivity'], cv = 10).mean() * 100)
print("sensitivity STD: ", cross_val_score(sel_alg, X, y, scoring=scoring['sensitivity'], cv = 10).std() * 100)
print("specificity: ", cross_val_score(sel_alg, X, y, scoring=scoring['specificity'], cv = 10).mean() * 100)
print("specificity STD: ", cross_val_score(sel_alg, X, y, scoring=scoring['specificity'], cv = 10).std() * 100)

# Save the trained model to a file
joblib.dump(est, f'{sel_alg}_model.joblib')
