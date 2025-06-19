# DeepFCMx-ELM

DeepFCMx is an FCM-based classification framework designed for medical image analysis, integrating CNNs with FCMs. The methodology begins with CNN training, where the feature maps are extracted from the last convolutional layer, capturing critical spatial and textural patterns within the medical images. These feature maps represent high-dimensional image descriptors, which serve as the primary input for the subsequent clustering stage. The feature maps are grouped based on the class labels of the corresponding images, and they are clustered, with the number of clusters optimized through experimentation to achieve optimal classification performance. To establish structured representations of image data, Euclidean distance is employed to compute similarities between extracted feature maps and the cluster centroids. This transformation allows the generation of a structured dataset, which serves as the input to the DeepFCMx classifier. In DeepFCMx-ELM, ELM is utilized to calculate the interconnections among concepts.

# Number of clusters
To set the number of clusters, please adjust the following value.
```
num_clusters= 3
```
# To set the path of the imaging folder

Ensure that your images have one of the following extensions

```
extensions = ['jpg', 'png', 'jpeg', 'tiff', '.tif','.TIFF', '.TIF']
```

Set the path to the image folder.
```
for ext in extensions:
    path_pattern = f"all_images/*.{ext}"
```
# Set the output names
Set the names of output classes
```
class0_name = 'normal'
class1_name='pathological'
```
# Set hyper-parameters and architecture of RGB-CNN
Set the number of pixel size, epochs, batch size, and architecture of RGB-CNN.
```
pixel_size=300
batch_size = 16
drop_rate = 0.1
num_epochs = 200

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
model.add(layers.Flatten())
model.add(layers.Dropout(drop_rate))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
```

# Set the parameters

DeepFCMx-ELM provides a transparent view of interconnections and their contributions to FCM's predictive accuracy.


To modify the learning rate, number of iterations and hidden units, adjust the following parameters accordingly
```
learning_rate = 0.01
num_iterations = 20
num_hidden_units=30
```
To alter the number of folds for k-fold cross validation, adjust the following parameter
```
#perform k-fold cross validation 
#Define K (num_folds) for k-fold cross validation
num_folds=10
```

# Usage

To use DeepFCMX-ELM, utilize the following code block:

```
best_fcm_weights = DeepFCMxELM(initial_weight_matrix, num_iterations, training_dataset, num_hidden_units,learning_rate)
```

# Training

The training process of DeepFCMx-ELM is demonstrated.

```

def DeepFCMxELM(initial_weights, num_iterations, input_features, learning_rate=0.001, lambda_reg=0.01):
    best_weights = initial_weights.copy()
    best_mse = float('inf')
    num_dimensions = initial_weights.shape[0]
    num_hidden_units = 20  # Number of hidden units in ELM, you can adjust this as needed
    # print(input_features)
    training_real_output = input_features[:, -1]  # Assuming the last column is the output

    lower_bounds = initial_weights - 0.1
    upper_bounds = initial_weights + 0.1

    for iteration in range(num_iterations):
        fcm_weights = update_fcm_weights(best_weights, input_features)
        fcm_weights = constrain_weights(fcm_weights, initial_weights, lower_bounds, upper_bounds)
        input_weights = np.random.uniform(size=(num_hidden_units, num_dimensions - 1), low=-1, high=1)
        hidden_layer_output = np.dot(input_features[:, :-1], input_weights.T)  # Exclude the last column for input features
        
        # Add bias term to hidden layer output
        hidden_layer_output_with_bias = np.hstack((hidden_layer_output, np.ones((hidden_layer_output.shape[0], 1))))
        
        # Compute output weights using Moore-Penrose pseudoinverse
        output_weights = np.linalg.pinv(hidden_layer_output_with_bias).dot(training_real_output)
        
        elm_output, _ = elm_predict_function(input_features[:, :-1], input_weights, output_weights)  # Exclude the last column for input features
        current_mse = mse_loss_with_regularization(training_real_output, elm_output, fcm_weights, initial_weights, lambda_reg)

        if current_mse < best_mse:
            best_mse = current_mse
            best_weights = fcm_weights
        
     
        
        best_weights = fcm_weights + learning_rate * ((training_real_output - elm_output) * elm_output * (1 - elm_output)).T @ input_features
        if current_mse < 0.8:
            break

    return best_weights

```

# Interpretation

This block of code will demonstrate the interconnections among concepts

```
plot_FCM_weight_matrix_graph(column_names, last_column_except_last)
```

# Interpretation

Grad-CAM provides interpretation of CNN predictions, highlighting critical regions that contribute to the model’s decision.
For NSCLC diagnosis with CT images
[https://www.thinkmind.org/library/EXPLAINABILITY/EXPLAINABILITY_2024/explainability_2024_1_60_10032.html](https://www.thinkmind.org/library/EXPLAINABILITY/EXPLAINABILITY_2024/explainability_2024_1_60_10032.html)
For CAD diagnosis with Polar Maps images
[https://www.mdpi.com/2076-3417/13/21/11953](https://www.mdpi.com/2076-3417/13/21/11953)


```
for filename in os.listdir(data_dir):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        file_path = os.path.join(data_dir, filename)
        image = cv2.imread(file_path)
        if image is None:
            print(f"File not found or unable to read: {file_path}")
            continue

        print("Processing:", file_path)
        if 'normal' in filename.lower():
            actual_class = 'normal'
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
            print("The model misclassified this instance as class1")
        else:
            print("The model predicted this instance as class2")
        
        if prediction == 0:
            predicted_class = 'class1'
            print("The model misclassified this instance as class1")
        else:
            predicted_class = 'class0'
            print("The model predicted this instance as class0")

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
```


## Dataset Description:

•	dataset.xlsx: It includes the clinical data in an array format with rows the number of instances and columns. For every instance, the id is provided.
•	suggested_weights: This Excel file is optional for FCM-GA. It includes the linguistic values for the input-output interconnections among FCM concepts provided by experts.

## Prerequisites:

-numpy
-time
-random
-pandas
-scikit-learn
-math
-statistics
-matplotlib
-networkx

## Supervisor

[Elpiniki Papageorgiou](https://emerald.uth.gr/personnel/)

## Contributors

[Anna Feleki](https://emerald.uth.gr/personnel/)
[Elpiniki Papageorgiou](https://emerald.uth.gr/personnel/)
[Ioannis Apostolopoulos](https://emerald.uth.gr/personnel/)
[Nikolaos Papandrianos](https://emerald.uth.gr/personnel/)
[Serafeim Moustakidis](https://emerald.uth.gr/personnel/)

## Assets

In the /assets folder, you'll find an example file for suggested_weights, and a visual representation of the interconnections among concepts in a FCM graph.
For suggested weights file, the following linguistic values must be provided, each representing a specific range:

For positive influence among concepts:
Very Weak (VW): [0,0.25]
Weak (W): [0.1,0.4]
Medium (M): [0.35,0.65]
Strong (S): [0.55,0.85]
Very Strong (VS): [0.75,1]

For negative influence among concepts:
-Very Weak (-VW): [-0.25,0]
-Weak (-W): [-0.4,-0.1]
-Medium (-M): [-0.65,-0.35]
-Strong (-S): [-0.85,-0.55]
-Very Strong (-VS): [-1,0.75]

Otherwise please provide the value :
random: [-1,1]
