#--- IMPORT DEPENDENCIES ------------------------------------------------------+
import math
import random
import time
from random import uniform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import KFold
import itertools
from statistics import mean
import os
import cv2
import networkx as nx
from openpyxl import load_workbook
import glob
from sklearn.model_selection import train_test_split
from fcm_functions import create_feature_maps
from keras import layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import cv2
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Model
from elm_predict import elm_predict_function
from fcm_functions import sig, DeepFCMxELM
import keras
from compute_mean_values import compute_mean_deviations
from suggested_weights_calculation import calculate_initial_weights
from calculate_deviations_file import calculate_deviation
from gradcam_file import GradCAM
#--- MAIN ---------------------------------------------------------------------+


###Define number of clusters
num_clusters= 3

print("training procedure")

data_dir = 'all_images/'

image_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.lower().endswith(('.tif', '.tiff', '.jpeg', '.png'))]
addrs=image_paths



num_dimensions= (num_clusters * 2 ) + 2

class0_name = 'normal'
class1_name='pathological'

class1_Feature_maps=[]
class0_Feature_maps=[]
# Number of classes
classes_num = 200
 

pixel_size=300

def read_and_process_image(list_of_images):
    X = []
    for img in list_of_images:
        image = cv2.imread(img)
        X.append(cv2.resize(image, (pixel_size, pixel_size), interpolation=cv2.INTER_CUBIC))

    # Define the output of each image to a different list
    y = [0 if class0_name in addr else 1 for addr in list_of_images]

    return X, y

random.shuffle(addrs)
X, y = read_and_process_image(addrs)
X=np.array(X)
y=np.array(y)
full_actual_output=y
print(X.shape)
 
channels=3

#The following values can be changed
# Batch size (should be a factor of 2.***4,8,16,32,64...***)
batch_size = 16

# Dropout rate
drop_rate = 0.1
 
# Number of epochs
num_epochs = 2

# save the weights of only the best model
filepath="med.weights.best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('plots/accuracy_loss'):
    os.makedirs('plots/accuracy_loss')
if not os.path.exists('plots/roc'):
    os.makedirs('plots/roc')
    
from sklearn.model_selection import KFold
#Define number of k-fold cross validation
num_folds=10

#Define number of classes
num_classes=2
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
nval = len(X_val)
ntest = len(X_test)

#Define architecture with number of nodes and layers for Convolutional Layers
model = Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu',input_shape=(pixel_size,pixel_size,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(drop_rate))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(drop_rate))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(drop_rate))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(drop_rate))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(drop_rate))

#Add extra convolutional layer
# model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(drop_rate))
model.add(layers.Flatten())
model.add(layers.Dropout(drop_rate))

#Define architecture with number of nodes and layers for Dense Layers
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(64,activation='relu'))

#Add extra fully connected layer
# model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# Compile the model (Different optimizer can be selected)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Different image data augmentation techniques can be selected
train_datagen = ImageDataGenerator(rescale=1./255, #Scale the image between 0 and 1
                                # rotation_range=90,
                                width_shift_range=0.1,
                                height_shift_range=0.1)
                                # zoom_range=0.1)
                                
                                           
# Set the validation dataset but not augmenting it (only rescaling it)
val_datagen = ImageDataGenerator(rescale=1./255)

# Set the test dataset but not augmenting it (only rescaling it)
test_datagen = ImageDataGenerator(rescale=1./255)

#Early stopping technique is applied to stop the training, when for '15' consecutive epochs the val loss has difference.
early_stopping = EarlyStopping(monitor='val_loss', patience=15, min_delta=1e-4, restore_best_weights = True)
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)

test_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=False)
# Model training
start = time.time()
history = model.fit(train_generator,
                                epochs=num_epochs,
                                validation_data = val_generator,
                                validation_steps = math.ceil(ntest / batch_size),
                                callbacks=[early_stopping])
end = time.time()
print('TIME = ', end-start, "sec.")

#For each fold the trained model is saved to the folder
model.save(f"model_rgb.keras")
#load pretrained model
from keras.models import load_model
model = keras.models.load_model("model_rgb.keras")

model.summary()

# get the output of the last fully connected layer
layer_name = 'dense_2' # choose the name of the last fully connected layer
feature_layer_model = Model(inputs=model.input,
                            outputs=model.get_layer(layer_name).output)

##############################################################################################
#########################################DeepFCMx------#######################################
##############################################################################################

dataset = create_feature_maps(addrs, num_clusters,pixel_size,feature_layer_model, class0_name, class1_name)


column_names = dataset.columns.tolist()


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


total_confusion_matrix = [[0, 0], [0, 0]]

mean_accuracies=[]
mean_losses=[]


#perform k-fold cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=9898)
limits_fold=[]
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
fold_precision=[]
fold_f1=[]
fold_recall=[]
fold_spec=[]
fold_sens=[]
# Iterate over each train-test split
import time
start = time.time()
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

    best_fcm_weights = DeepFCMxELM(initial_weight_matrix, num_iterations, training_dataset, num_hidden_units,learning_rate)
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
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    # Safe defaults
    sensitivity = 0
    specificity = 0
    precision_score = 0

    if accuracy_value != 100 and (cm[0,0]+cm[0,1]) > 0 and (cm[1,0]+cm[1,1]) > 0:
        sensitivity1 = cm[0,0] / (cm[0,0] + cm[0,1])
        specificity1 = cm[1,1] / (cm[1,0] + cm[1,1])

        sensitivity = sensitivity1 * 100
        specificity = specificity1 * 100

        conf_accuracy = (TP + TN) / (TP + TN + FP + FN)
        conf_misclassification = 1 - conf_accuracy

        conf_sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        conf_specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

        def precision(TP, FP):
            return TP / (TP + FP) if (TP + FP) != 0 else 0

        precision_score = precision(TP, FP)

    # Store metrics
    sens.append(sensitivity)
    spec.append(specificity)
    prec.append(precision_score * 100)


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


####Grad-CAM methodology
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
            print("The model misclassified this instance as " + class0_name)
        else:
            print("The model predicted this instance as " + class1_name)
            
        if prediction == 0:
            predicted_class = 'class1'
            print("The model misclassified this instance as " + class1_name)
        else:
            predicted_class = class0_name
            print("The model predicted this instance as" + class0_name)

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
