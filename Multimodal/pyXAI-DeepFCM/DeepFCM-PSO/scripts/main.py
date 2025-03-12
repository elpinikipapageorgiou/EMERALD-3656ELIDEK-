#Import libraries
from sklearn.model_selection import KFold
import math
from sklearn.metrics import accuracy_score, confusion_matrix
import time
from sklearn import metrics
from statistics import mean
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras

######Import functions
from particle_functions import DeepFCM_PSO, sig
from deviations import calculate_deviation
from compute_mean_values import compute_mean_deviations
from plot_fcm import plot_FCM_weight_matrix_graph
from gradcam_file import GradCAM
#--- MAIN ---------------------------------------------------------------------+

###################################################################
###################################################################
################# Concat CNN predictions with tabular data ########
###################################################################
###################################################################

# Ensure reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Define pixel size for rows and columns
pixel_size = 300

class0_name='normal'
class1_name='pathological'

# Function to read and process images
def read_and_process_image(list_of_images):
    X = []
    for img in list_of_images:
        image = cv2.imread(img)
        X.append(cv2.resize(image, (pixel_size, pixel_size), interpolation=cv2.INTER_CUBIC))

    # Define the output of each image to a different list
    y = [0 if class0_name in addr else 1 for addr in list_of_images]

    return X, y

# Image directory
data_dir = 'images/'

# Collect all image files
image_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.lower().endswith(('.tif', '.tiff', '.jpeg', '.png'))]
random.shuffle(image_files)

# Read and process images
X, y = read_and_process_image(image_files)
X = np.array(X)
y = np.array(y)

# Normalize pixel values
X = X / 255.0

# Split data
X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(X, y, image_files, test_size=0.20, random_state=42)

# Define model architecture
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(pixel_size, pixel_size, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.1),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.1),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.1),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.1),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.1),
    Flatten(),
    Dropout(0.1),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=15, min_delta=1e-4, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=200, validation_split=0.2, callbacks=[early_stopping])

model.save(f"model_rgb.keras")


###Grad-CAM


model = keras.models.load_model("model_rgb.keras")


output_folder = 'gradcam_results'

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List of common image extensions (Add the preffered extension)
image_extensions = ['.tiff', '.tif', '.TIFF', '.jpeg', '.jpg', '.png']
last_conv_layer_name = 'conv2d_2'  # Replace 'conv2d_2' with your actual last convolutional layer name

# Iterate over all files in the directory that match the extensions

for filename in os.listdir(data_dir):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        file_path = os.path.join(data_dir, filename)
        image = cv2.imread(file_path)
        if image is None:
            print(f"File not found or unable to read: {file_path}")
            continue

        print("Processing:", file_path)
        if class0_name in filename.lower():
            actual_class = class0_name
        else:
            actual_class = class1_name
        # Preprocess the image
        image_resized = cv2.resize(image, (pixel_size, pixel_size))
        image_resized = image_resized.astype('float32') / 255
        image_resized = np.expand_dims(image_resized, axis=0)

        # Predict the class
        preds = model.predict(image_resized)
        prediction = (preds > 0.5).astype(int)

        if prediction == 0:
            print("The model misclassified this instance as" + class0_name)
        else:
            print("The model predicted this instance as " + class1_name)
            
        if prediction == 0:
            predicted_class = class0_name
            print("The model predicted this instance as "+ class0_name)
        else:
            predicted_class = 'class1'
            print("The model predicted this instance as " + class1_name)

        # Define the GradCAM instance with the manually set layer name
        icam = GradCAM(model, np.argmax(preds[0]), layerName=last_conv_layer_name)
        heatmap = icam.compute_heatmap(image_resized)
        heatmap_resized = cv2.resize(heatmap, (pixel_size, pixel_size))

        # Overlay the heatmap onto the original image
        (heatmap, output) = icam.overlay_heatmap(heatmap_resized, image, alpha=0.5)

        # Convert images to RGB for display
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        
        # Concatenate heatmap and original image side by side
        concatenated = np.concatenate((image, heatmap_rgb), axis=1)

        # Save the concatenated image
        output_filename = os.path.join(output_folder, f"gradcam_actual{actual_class}_pred_{predicted_class}_{os.path.splitext(filename)[0]}.png")
        cv2.imwrite(output_filename, concatenated)

        print(f"Saved concatenated image at: {output_filename}")




# Make predictions for the entire dataset
all_predictions = model.predict(X).flatten()

# Convert probabilities to binary predictions (0 or 1)
cnn_predictions = [1 if p > 0.5 else 0 for p in all_predictions]

# Generate IDs for all instances (assuming the order in X corresponds to image_files)
ids = [os.path.basename(file)[:4] for file in image_files]

# Create a DataFrame with IDs and CNN predictions
predictions_df = pd.DataFrame({
    'id': ids,
    'CNN_Prediction': cnn_predictions
})

# Save to an Excel file
output_file_name = 'cnn_predictions_all_instances.xlsx'
predictions_df.to_excel(output_file_name, index=False)


#Read excel dataset
excel_dataset=pd.read_excel("dataset.xlsx", engine='openpyxl')



# # apply normalization techniques by Column
#How to apply normalization to the columns Age and BMI
columns_to_normalize = ['column1', 'column2']  # Add any other columns you need to normalize

for column in columns_to_normalize:
    excel_dataset[column] = (excel_dataset[column] - excel_dataset[column].min()) / (excel_dataset[column].max() - excel_dataset[column].min())

#Concat CNN's predictions with clinical data
excel_dataset.columns.values[0] = 'id'
cnn_outputs = pd.read_excel('cnn_predictions_all_instances.xlsx', engine='openpyxl')
list_of_ids_cnn_outputs = cnn_outputs.iloc[:, 0].tolist()
cnn_outputs = pd.DataFrame(cnn_outputs)
cnn_outputs.iloc[:, 0] = list_of_ids_cnn_outputs
cnn_outputs = cnn_outputs.rename(columns={cnn_outputs.columns[0]: 'ids'})
print(excel_dataset, cnn_outputs)
#merge clinical dataset with CNN predictions
dataset = pd.merge(excel_dataset, cnn_outputs, left_on='id', right_on='ids', how='inner')

# Remove the 'age' column and append it to the end of the DataFrame
age_column = dataset.pop('id')
age_column = dataset.pop('ids')
output = dataset.pop('output')
dataset = dataset.assign(output=output)

num_dimensions= dataset.shape[1] 
###################################################################
###################################################################
################# --End of Concatenation--#########################
###################################################################
###################################################################


#Fill missing values with the method bfill
dataset.fillna(method="bfill", inplace=True)

#Store the column names
column_names = dataset.columns.tolist()

# The user should specify if they want to provide suggested initial weights by loading from an external file.
# Set `suggested_weights` to None by default, then load the weights if a file is specified.
suggested_weights=None
# suggested_weights = pd.read_excel("suggested_weights.xlsx", engine='openpyxl')


##Define number of particles
num_particles = 40

#define number of epochs
epoch = 20

#Perform k-fold cross validation
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

    #call the training function for each train split
    best_position,concept_evolution = DeepFCM_PSO(training_dataset, num_dimensions=num_dimensions,
                                                                          num_particles=num_particles, maxiter=epoch,
                                                                          suggested_weights=suggested_weights)
    end = time.time()
    # Plotting concept evolution
    plt.figure(figsize=(15, 10))
    epoch_range = range(1, epoch + 1)  # Create an array for the x-axis representing epochs
    for j in range(num_dimensions):
        plt.plot(epoch_range, concept_evolution[:, j], label=f'Concept {j+1}')
    plt.xlabel('Epochs')
    plt.ylabel('Concept Value')
    plt.title(f'Evolution of Concept Values Over Epochs (Fold {fold})')
    plt.legend()
    plt.legend(loc='upper right')
    # Save the plot as an image file
    filename = f'fold_{fold}_inference_plot.png'  # Define your filename here
    plt.savefig(filename)  # Save the plot
    plt.close()  # Close the plot to free memory


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
                
    #perform testing procedure
    error=[]
    sum_temp=0
    #from the testing dataset calculate the predcited values and compare with the actual output
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

    #seperate the predicted values to the two columns
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

    
    #compute performance metrics
    A=(accuracy_score(testing_actual_output, testing_last_element_fcm_output.round())*100)
    acc.append(accuracy_score(testing_actual_output, testing_last_element_fcm_output.round())*100)

    err.append(metrics.mean_absolute_error(testing_actual_output, testing_last_element_fcm_output))
    cm = confusion_matrix(testing_actual_output, testing_last_element_fcm_output)
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

        print("-----------------")

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

        # convert the NumPy array to a pandas DataFrame
        #each column should be inserted to this list
        df = pd.DataFrame(best_position, columns=column_names)

    # write the best position weight matrix to an Excel file for each fold
    df.to_excel(f'data{fold}.xlsx', index=False)


#Print performance metrics
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

print("Sum of Confusion Matrices:")
print(sum_matrix)

print("\nSensitivity")
sens = [x for x in sens if not math.isnan(x)]
print(sens)
print(np.mean(sens))
sens_deviation = calculate_deviation(sens)
print("sens_deviation")
print(sens_deviation)

print("\nSpecificity")
spec = [x for x in spec if not math.isnan(x)]
print(spec)
print(np.mean(spec))
spec_deviation = calculate_deviation(spec)
print("spec_deviation")
print(spec_deviation)

print("\n Precision")
prec = [x for x in prec if not math.isnan(x)]
print(prec)

print(np.mean(prec))
prec_deviation = calculate_deviation(prec)
print("prec_deviation")
print(prec_deviation)
print("-----------\n\n------------")
      

compute_mean_deviations(num_dimensions, fold, column_names)
last_column_except_last = [row[-1] for row in best_position[:-1]]
plot_FCM_weight_matrix_graph(column_names, last_column_except_last)