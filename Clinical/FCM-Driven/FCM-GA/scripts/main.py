#Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

###Import functions
from generate_population import generate_population_from_excel
from fcm_functions import FCM_GA, predict_fcm
from deviation import calculate_deviation
from compute_mean_values import compute_mean_deviations
from plot_fcm import plot_FCM_weight_matrix_graph
#--- MAIN ---------------------------------------------------------------------

#define population size
pop_size= 100 

#define number of generations
num_generations = 100

#define mutation rate
mutation_rate = 0.1



#read excel dataset
dataset=pd.read_excel("dataset.xlsx", engine='openpyxl')

num_dimensions= dataset.shape[1] 

column_names = dataset.columns.tolist()

#Fill missing values (Optional)
dataset.fillna(method="bfill", inplace=True)

# # apply normalization techniques by Column
columns_to_normalize = ['column1', 'column2']  # Add any other columns you need to normalize

for column in columns_to_normalize:
    dataset[column] = (dataset[column] - dataset[column].min()) / (dataset[column].max() - dataset[column].min())


print(dataset)

#Excel file with interconnections among concepts
# The user should specify if they want to provide suggested initial weights by loading from an external file.
# Set `suggested_weights` to None by default, then load the weights if a file is specified.

excel_file_path=None
# excel_file_path = "suggested_weights.xlsx"

population = generate_population_from_excel(pop_size, num_dimensions, excel_file_path)


num_folds=10

fold=0
acc=[]
loss=[]
recall=[]
precision=[]
f1=[]
cm_sum=[]
sens=[]
spec=[]

#define number of iterations
num_iterations=15

#K-fold cross validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
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
    

    testing_dataset = pd.DataFrame(testing_dataset)
    testing_dataset = testing_dataset.values
  
    # Get the best FCM from the final population
    best_fcm = FCM_GA(pop_size, mutation_rate, num_dimensions, training_dataset, excel_file_path)

    np.fill_diagonal(best_fcm, 0)
    num_rows_to_add = 1

    # Creating a row of zeros
    zeros_row = np.zeros((num_rows_to_add, best_fcm.shape[1]))

    # Adding rows of zeros to the original array
    best_fcm1 = np.vstack((best_fcm, zeros_row))
    
    #the columns names should be inserted as a list
    df = pd.DataFrame(best_fcm1, columns=column_names)

    # Save DataFrame to Excel
    df.to_excel(f'data{fold}.xlsx', index=False)

    # Evaluate performance metrics on testing data
    predictions = predict_fcm(best_fcm, testing_dataset)
    testing_dataset = np.array(testing_dataset)
    true_labels = testing_dataset[:, -1]

    temporary_value_results=predictions


    limits_acc=[]
    limits= np.arange(0.1, 0.99, 0.01).tolist()

    steady_value_predicted_results=temporary_value_results
    for i in limits:

        temporary_value_results = steady_value_predicted_results > i

        true_labels=np.array(true_labels)
        temporary_value_results=(np.array(temporary_value_results))

        limits_acc.append(accuracy_score(true_labels, temporary_value_results.round())*100)


    max_value = max(limits_acc)
    index = limits_acc.index(max_value)

    import matplotlib.pyplot as plt
    predictions = predictions > limits[index]

    true_labels=np.array(true_labels)
    predictions=(np.array(predictions))


    acc.append(accuracy_score(true_labels, predictions)*100)
    loss.append(metrics.mean_absolute_error(true_labels, predictions))
    recall.append(recall_score(true_labels, predictions, average='weighted')*100)
    f1.append(f1_score(true_labels, predictions, average='weighted')*100)
    precision.append(precision_score(true_labels, predictions, average='weighted')*100)
    A = accuracy_score(true_labels, predictions)*100
    cm = confusion_matrix(true_labels, predictions)
    cm_sum.append(cm)

    class_counts = cm.sum(axis=1)
    accuracies = [0 if count == 0 else cm[i, i] / count for i, count in enumerate(class_counts)]

    if(A!=100):
        TP = cm[1][1]
        TN = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]

        sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])

        sens.append(sensitivity1*100)

        specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])

        spec.append(specificity1*100)

print("\n\n\n")
print("-------------end of kfold------------")
from statistics import mean
print("Accuracies")
print(acc)
print(mean(acc))
acc_deviation = calculate_deviation(acc)
print("acc_deviation")
print(acc_deviation)



print("\n\nError")
print(loss)
print(mean(loss))
err_deviation = calculate_deviation(loss)
print("err_deviation")
print(err_deviation)

print("\n\Sensitivity")
print(sens)
print(mean(sens))
sens_deviation = calculate_deviation(sens)
print("sens_deviation")
print(sens_deviation)

print("\n\Specificity")
print(spec)
print(mean(spec))
spec_deviation = calculate_deviation(spec)
print("spec_deviation")
print(spec_deviation)


print("\n\Precision")
print(precision)
print(mean(precision))
precision_deviation = calculate_deviation(precision)
print("precision_deviation")
print(precision_deviation)

compute_mean_deviations(num_dimensions, fold, column_names)
last_column_except_last = [row[-1] for row in best_fcm[:-1]]
plot_FCM_weight_matrix_graph(column_names, last_column_except_last)