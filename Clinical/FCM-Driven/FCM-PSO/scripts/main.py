#--- IMPORT DEPENDENCIES ------------------------------------------------------+
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from statistics import mean
import matplotlib.pyplot as plt

######Import functions
from particle_functions import FCM_PSO, sig
from deviations import calculate_deviation
from compute_mean_values import compute_mean_deviations
from plot_fcm import plot_FCM_weight_matrix_graph
#--- MAIN ---------------------------------------------------------------------

#read_tabular_data
dataset=pd.read_excel("dataset.xlsx", engine='openpyxl')

num_dimensions= dataset.shape[1] 

column_names = dataset.columns.tolist()

## The fillna() function in this code replaces missing values (NaNs) in the dataset with valid values found earlier in the dataset, using the backfill method
dataset.fillna(method="bfill", inplace=True)

# # apply normalization techniques by Column
columns_to_normalize = ['column1', 'column2']  # Add any other columns you need to normalize

for column in columns_to_normalize:
    dataset[column] = (dataset[column] - dataset[column].min()) / (dataset[column].max() - dataset[column].min())


# The user should specify if they want to provide suggested initial weights by loading from an external file.
# Set `suggested_weights` to None by default, then load the weights if a file is specified.
suggested_weights=None
# suggested_weights = pd.read_excel("suggested_weights.xlsx", engine='openpyxl')

##Define number of particles
num_particles = 40

#Define number of epochs
epoch=30

##Count time
start = time.time()

#perform k-fold cross validation 
#Define K (num_folds) for k-fold cross validation
num_folds=10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

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

    # apply minimization function to each training split
    best_position, concept_evolution = FCM_PSO(training_dataset,  num_dimensions=num_dimensions, num_particles=num_particles, maxiter=epoch, suggested_weights=suggested_weights)
    end = time.time()
    # Plotting concept evolution
    # plt.figure(figsize=(15, 10))
    # epoch_range = range(1, epoch + 1)  # Create an array for the x-axis representing epochs
    # for j in range(num_dimensions):
    #     plt.plot(epoch_range, concept_evolution[:, j], label=f'Concept {j+1}')
    # plt.xlabel('Epochs')
    # plt.ylabel('Concept Value')
    # plt.title(f'Evolution of Concept Values Over Epochs (Fold {fold})')
    # plt.legend()
    # plt.legend(loc='upper right')
    # # Save the plot as an image file
    # filename = f'fold_{fold}_inference_plot.png'  # Define your filename here
    # plt.savefig(filename)  # Save the plot
    # plt.close()  # Close the plot to free memory
    for i in range(0,num_dimensions):
        for j in range(0,num_dimensions):
            if(i==j):
                continue
            else:
                # adjust maximum position if necessary
                if best_position[i][j]>1:
                    best_position[i][j]=1

                # # adjust minimum position if neseccary
                if best_position[i][j]<-1:
                    best_position[i][j]=-1

    #construct testing dataset
    sum_temp=0

    testing_last_element_fcm_output=[]
    best_position=np.vstack(best_position)
    predicted_results=[None]*num_dimensions
    for testing_row in testing_dataset:

        for i in range(0,num_dimensions):
            for j in range(0,num_dimensions):
                if(i==j):
                    continue
                else:
                    sum_temp=sum_temp+best_position[j][i]*testing_row[j]

            predicted_results[i]=sum_temp
            predicted_results[i] = sig(predicted_results[i])

            sum_temp=0

        testing_last_element_fcm_output.append((predicted_results)[-1])
    testing_actual_output = testing_dataset[:, -1]

    testing_last_element_fcm_output = np.vstack(testing_last_element_fcm_output)

    testing_last_element_fcm_output = testing_last_element_fcm_output[:, -1]

    temporary_value_results=testing_last_element_fcm_output

    #find the best limit to seperate testing outputs to classes
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

    testing_actual_output=np.array(testing_actual_output)
    testing_last_element_fcm_output=(np.array(testing_last_element_fcm_output))

    #calculate performance metrics (accuracy, error, sensitivity, specificity, precision)
    A=(accuracy_score(testing_actual_output, testing_last_element_fcm_output.round())*100)
    acc.append(accuracy_score(testing_actual_output, testing_last_element_fcm_output.round())*100)


    err.append(metrics.mean_absolute_error(testing_actual_output, testing_last_element_fcm_output))
    cm = confusion_matrix(testing_actual_output, testing_last_element_fcm_output)
    cm_sum.append(cm)
    class_counts = cm.sum(axis=1)
    accuracies = [0 if count == 0 else cm[i, i] / count for i, count in enumerate(class_counts)]

    # convert the best weight matrix to a pandas DataFrame
    df = pd.DataFrame(best_position, columns=column_names)

    # write the DataFrame to an Excel file for each fold
    df.to_excel(f'data{fold}.xlsx', index=False)

    if(A!=100):
        TP = cm[1][1]
        TN = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]

        sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])

        sens.append(sensitivity1*100)

        specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])

        spec.append(specificity1*100)

        # calculate accuracy
        conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))

        # calculate mis-classification
        conf_misclassification = 1- conf_accuracy

        # calculate the sensitivity
        conf_sensitivity = (TP / float(TP + FN))
        # calculate the specificity
        conf_specificity = (TN / float(TN + FP))


        def precision(TP, FP):
            return TP / (TP + FP)
        # calculate precision
        precision_score = precision(TP, FP)
        prec.append(precision_score*100)


#print the performance metrics
print("\n\n\n")
print("-------------end of kfold------------")

print("Accuracies")
print(acc)
print(mean(acc))


acc_deviation = calculate_deviation(acc)
print("acc_deviation")
print(acc_deviation)



print("\n\nError")
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


compute_mean_deviations(num_dimensions, fold, column_names)
last_column_except_last = [row[-1] for row in best_position[:-1]]
plot_FCM_weight_matrix_graph(column_names, last_column_except_last)