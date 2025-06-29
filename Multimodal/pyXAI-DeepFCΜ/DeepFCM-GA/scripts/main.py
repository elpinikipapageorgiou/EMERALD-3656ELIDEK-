#Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import keras
###Import functions
from generate_population import generate_population_from_excel
from fcm_functions import DeepFCM_GA, predict_fcm
from deviation import calculate_deviation
from compute_mean_values import compute_mean_deviations
from plot_fcm import plot_FCM_weight_matrix_graph
from gradcam_file import GradCAM



###################################################################
###################################################################
################# Concat CNN predictions with tabular data ########
###################################################################
###################################################################

# Ensure reproducibility
random.seed(42)
np.random.seed(42)

class0_name='normal'
class1_name='pathological'

pixel_size = 300
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

num_epochs=200
batch_size=32

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
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])
model.save(f"model_rgb.keras")

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
#read the intial linguistic values provided by nuclear experts for the initialization of interconnections among input-output concepts

#--- MAIN ---------------------------------------------------------------------

#define population size
pop_size= 40 

#define number of generations
num_generations = 150

#define crossover rate
crossover_rate = 0.8

#define mutation rate
mutation_rate = 0.08

#define number of iterations
num_iterations=40

num_folds=10

excel_file_path=None
# excel_file_path = "suggested_weights.xlsx"

population = generate_population_from_excel(pop_size, num_dimensions, excel_file_path)


fold=0
acc=[]
loss=[]
recall=[]
precision=[]
f1=[]
cm_sum=[]
sens=[]
spec=[]
kf = KFold(n_splits=num_folds, shuffle=True)
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
    best_fcm = DeepFCM_GA(pop_size, crossover_rate, mutation_rate, num_dimensions, training_dataset, excel_file_path)

    # print("Best FCM:", best_fcm.shape)
    np.fill_diagonal(best_fcm, 0)
    num_rows_to_add = 1

    # Creating a row of zeros
    zeros_row = np.zeros((num_rows_to_add, best_fcm.shape[1]))

    # Adding rows of zeros to the original array
    best_fcm1 = np.vstack((best_fcm, zeros_row))
    df = pd.DataFrame(best_fcm1, columns=column_names)


    # Save DataFrame to Excel

    df.to_excel(f'data{fold}.xlsx', index=False)

    # Evaluate performance metrics on testing data
    # print(testing_set1)
    predictions = predict_fcm(best_fcm, testing_dataset)
    testing_dataset = np.array(testing_dataset)
    true_labels = testing_dataset[:, -1]


    # print(predictions)
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
    # print(limits[index])

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
    # print(cm)

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
last_column_except_last = [row[-1] for row in best_fcm1[:-1]]
plot_FCM_weight_matrix_graph(column_names, last_column_except_last)


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
        if 'class0' in filename.lower():
            actual_class = 'class0'
        else:
            actual_class = 'class1'
        # Preprocess the image
        image_resized = cv2.resize(image, (pixel_size, pixel_size))
        image_resized = image_resized.astype('float32') / 255
        image_resized = np.expand_dims(image_resized, axis=0)

        # Predict the class
        preds = model.predict(image_resized)
        prediction = (preds > 0.5).astype(int)

        if prediction == 0:
            predicted_class = 'class0'
            print("The model predicted this instance as class0.")
        else:
            predicted_class = 'class1'
            print("The model predicted this instance as class1.")

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
