import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import KFold
from openpyxl import load_workbook
from statistics import mean

##Import Functions
from suggested_weights_calculation import calculate_initial_weights
from calculate_deviations_file import calculate_deviation
from fcm_functions import sig, FCM_ELM
from compute_mean_values import compute_mean_deviations
from plot_fcm import plot_FCM_weight_matrix_graph

#Please enter the dataset
dataset=pd.read_excel("dataset.xlsx", engine='openpyxl')
column_names = dataset.columns.tolist()

#Please enter the selected line to impute the missing values
dataset.fillna(method="bfill", inplace=True)

# # apply normalization techniques by Column
columns_to_normalize = ['column1', 'column2']  # Add any other columns you need to normalize

for column in columns_to_normalize:
    dataset[column] = (dataset[column] - dataset[column].min()) / (dataset[column].max() - dataset[column].min())


dataset.to_excel('dataset.xlsx', index=False)
from sklearn.utils import shuffle
dataset = shuffle(dataset)

print(dataset)

num_dimensions = len(dataset.columns)  # Assuming the last column is the output

num_hidden_units = num_dimensions

#Values that can be altered
learning_rate = 0.01
num_iterations = 20
num_hidden_units=30

#In case user uploads weights in the form of fuzzy sets for the initialization of interconnections
suggested_weights=None
# suggested_weights = pd.read_excel("suggested_weights.xlsx", engine='openpyxl')

initial_weight_matrix=calculate_initial_weights(num_dimensions, suggested_weights)


dataset = pd.DataFrame(dataset)


print(dataset.shape)

print(dataset)


#perform k-fold cross validation you can change the value
folds=10


kf = KFold(n_splits=folds, shuffle=True, random_state=42)

fold=0
acc=[]
err=[]
best_matrix=[]
concepts=[]
acc=[]
err=[]
best_matrix=[]
concepts=[]
class_accuracies0 =[]
class_accuracies1 =[]
c_matrices=[]
recalls=[]
best_positions=[]
cm_sum = []
sens=[]
spec=[]
prec=[]
limits_fold=[]
# Iterate over each train-test split
for train_index, test_index in kf.split(dataset):
    fold+=1


    print("\n\n************************************")
    print(fold)
    print("---------fold------------")
    print("************************************\n\n")

    # Split train-test
    training_dataset, testing_dataset=dataset.iloc[train_index], dataset.iloc[test_index]

    training_dataset= np.array(training_dataset)
    testing_dataset= np.array(testing_dataset)


    # Optimize FCM weights
    best_fcm_weights = FCM_ELM(initial_weight_matrix, num_iterations, training_dataset, num_hidden_units,learning_rate)


    for i in range(0,num_dimensions):
        for j in range(0,num_dimensions):
            if(i==j):
                continue
            else:
                # adjust maximum position if necessary
                if best_fcm_weights[i][j]>1:
                    # print(best_fcm_weights[i][j])
                    best_fcm_weights[i][j]=1
                    # print(best_fcm_weights[i][j])

                # # adjust minimum position if neseccary
                if best_fcm_weights[i][j]<-1:
                    best_fcm_weights[i][j]=-1

    sum_temp=0
    testing_last_element_fcm_output=[]
    predicted_results=[None]*num_dimensions
    for testing_row in testing_dataset:

        for i in range(0,num_dimensions):
            for j in range(0,num_dimensions):
                if(i==j):
                    continue
                else:
                    sum_temp=sum_temp+best_fcm_weights[j][i]*testing_row[j]

            predicted_results[i]=sum_temp
            predicted_results[i] = sig(predicted_results[i])

            sum_temp=0


        testing_last_element_fcm_output.append((predicted_results)[-1])
    testing_actual_output = testing_dataset[:, -1]


    testing_last_element_fcm_output = np.vstack(testing_last_element_fcm_output)

    testing_last_element_fcm_output = testing_last_element_fcm_output[:, -1]


    temporary_value_results=testing_last_element_fcm_output


    limits_acc=[]
    limits= np.arange(0.1, 0.99, 0.01).tolist()

    steady_value_predicted_results=temporary_value_results
    for i in limits:

        temporary_value_results = steady_value_predicted_results > i

        testing_actual_output=np.array(testing_actual_output)
        temporary_value_results=(np.array(temporary_value_results))

        limits_acc.append(accuracy_score(testing_actual_output, temporary_value_results.round())*100)


    max_value = max(limits_acc)
    index = limits_acc.index(max_value)

    testing_last_element_fcm_output = testing_last_element_fcm_output > limits[index]
    limits_fold.append(limits[index])

    testing_actual_output=np.array(testing_actual_output)
    testing_last_element_fcm_output=(np.array(testing_last_element_fcm_output))

    accuracy_value=(accuracy_score(testing_actual_output, testing_last_element_fcm_output.round())*100)
    acc.append(accuracy_value)

    error=(metrics.mean_absolute_error(testing_actual_output, testing_last_element_fcm_output))
    err.append(error)
    cm = confusion_matrix(testing_actual_output, testing_last_element_fcm_output)

    cm_sum.append(cm)

    class_counts = cm.sum(axis=1)
    accuracies = [0 if count == 0 else cm[i, i] / count for i, count in enumerate(class_counts)]

    if(accuracy_value!=100):
        TP = cm[1][1]
        TN = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]

        sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])

        sensitivity=(sensitivity1*100)

        specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])

        specificity=(specificity1*100)

        conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))

        conf_misclassification = 1- conf_accuracy

        conf_sensitivity = (TP / float(TP + FN))
        conf_specificity = (TN / float(TN + FP))


        def precision(TP, FP):
            return TP / (TP + FP)
        precision_score = precision(TP, FP)
    sens.append(sensitivity)
    spec.append(specificity)
    prec.append(precision_score*100)

    # convert the NumPy array to a pandas DataFrame
    # data = best_fcm_weights[1:]  # Extract the rest of the rows for the actual data

    # Create the DataFrame with dynamic column names
    df = pd.DataFrame(best_fcm_weights, columns=column_names)

    df.to_excel(f'data{fold}.xlsx', index=False)


#print the performance metrics
print("\n\n\n")
print("-------------end of kfold------------")

print("Accuracies")
print(acc)
print(mean(acc))


acc_deviation = calculate_deviation(acc)
print("acc_deviation")
print(acc_deviation)

print("\nError")
print(err)
print(mean(err))
err_deviation = calculate_deviation(err)
print("err_deviation")
print(err_deviation)

sum_matrix = np.sum(cm_sum, axis=0)

print("\nSum of Confusion Matrices:")
print(sum_matrix)

print("\nSensitivity")
print(sens)
print(np.mean(sens))
sens_deviation = calculate_deviation(sens)
print("sens_deviation")
print(sens_deviation)


print("\nSpecificity")
print(spec)
print(np.mean(spec))
spec_deviation = calculate_deviation(spec)
print("spec_deviation")
print(spec_deviation)

print("\n Precision")
print(prec)
print(np.mean(prec))
prec_deviation = calculate_deviation(prec)
print("prec_deviation")
print(prec_deviation)



print("\n limits_fold")
print(limits_fold)
print(np.mean(limits_fold))
prec_deviation = calculate_deviation(limits_fold)
print("limits_fold")
print(prec_deviation)

compute_mean_deviations(num_dimensions, fold, column_names)
last_column_except_last = [row[-1] for row in best_fcm_weights[:-1]]
plot_FCM_weight_matrix_graph(column_names, last_column_except_last)