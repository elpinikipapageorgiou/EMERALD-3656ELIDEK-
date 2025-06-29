# DeepFCM-ELM

This is the official implementation of
[https://dl.acm.org/doi/full/10.1145/3716554.3716615](https://dl.acm.org/doi/full/10.1145/3716554.3716615)

# Paper Abstract
Early detection of Coronary Artery Disease (CAD) and Non-Small Cell Lung Cancer (NSCLC) is crucial for improving patient outcomes. In this study, RGB-CNN (Convolutional Neural Network) was implemented, and trained from scratch using Polar Maps for CAD diagnosis and Computed Tomography (CT) images for NSCLC diagnosis. The CNN predictions were then integrated with clinical data into a Fuzzy Cognitive Map (FCM) classifier for each type of diagnosis. Nuclear medicine experts provided linguistic values in the form of fuzzy sets to define the relationships between input and output concepts, which were later converted into interval values. Extreme Learning Machine (ELM) and Genetic Algorithm (GA) were applied to the FCM learning process to refine the interconnections based on expert knowledge. To ensure the robustness of the results, 10-fold cross-validation was employed. The DeepFCM-ELM model demonstrated superior performance, achieving 80.4%±4.97% accuracy for CAD diagnosis, and 91.9%±3.07% for NSCLC diagnosis using CT images. Heatmaps were generated to interpret CNN predictions by highlighting pathological regions. These heatmaps were then used in GPT, along with DeepFCM weights, CNN, and DeepFCM prediction and input clinical values, employing Natural Language Generation to translate DeepFCM results into human-readable language, enhancing the model's overall explainability. All these techniques have been integrated into a Medical Decision Support System (MDSS) designed to effectively manage both medical classification challenges.

# Multimodal approach (Clinical + Imaging Data)

This multimodal approach enhances interpretability by integrating both clinical and imaging data, providing a transparent view of interconnections and their contributions to DeepFCM's predictive accuracy.

# Define class labels

class0_name='normal'

class1_name='pathological'

# To set the path of the imaging folder

Specify the input size (in pixels) for resizing all images:

pixel_size = 300

Ensure that your images have one of the following extensions

```
image_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.lower().endswith(('.tif', '.tiff', '.jpeg', '.png'))]
```

Set the number of epochs, batch size, and architecture of RGB-CNN.

epochs = 200

batch_size = 32

# CNN Architecture (RGB-CNN)
```
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
```
# Clinical Data Normalization

Select which columns to be normalized with the Min-max normalization

columns_to_normalize = ['column1', 'column2']  # Add any other columns you need to normalize

# Define hyper-parameters for DeepFCM-PSO

learning_rate = 0.01

num_iterations = 20

num_hidden_units = 30

# Expert knowledge (if provided)

If expert knowledge is provided in the form of fuzzy sets, the user must define the Excel file.

suggested_weights = pd.read_excel("suggested_weights.xlsx", engine='openpyxl')

# Usage

To use DeepFCM-ELM, utilize the following code block:

```
best_fcm_weights = DeepFCM_ELM(initial_weight_matrix, num_iterations, training_dataset, learning_rate)
```

# Training

The training process of DeepFCM-ELM is demonstrated.

```

def DeepFCM_ELM(initial_weights, num_iterations, input_features, learning_rate=0.1, lambda_reg=0.01):
    best_weights = initial_weights.copy()
    best_mse = float('inf')
    num_dimensions = initial_weights.shape[0]
    num_hidden_units = 30  # Number of hidden units in ELM, you can adjust this as needed

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
  
        elm_output, _ = elm_predict(input_features[:, :-1], input_weights, output_weights)  # Exclude the last column for input features
        current_mse = mse_loss_with_regularization(training_real_output, elm_output, fcm_weights, initial_weights, lambda_reg)

        if current_mse < best_mse:
            best_mse = current_mse
            best_weights = fcm_weights
  
        if current_mse < 0.8:
            break
  
        best_weights = fcm_weights + learning_rate * ((training_real_output - elm_output) * elm_output * (1 - elm_output)).T @ input_features[:, :-1]

    return best_fcm
```

# Interpretation

This block of code will demonstrate the interconnections among concepts

```
plot_FCM_weight_matrix_graph(column_names, last_column_except_last)
```
![Plot](assets/DeepFCM_ELM_CAD.jpg)
Additionally, Grad-CAM provides interpretation of CNN predictions, highlighting critical regions that contribute to the model’s decision.

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

•	dataset.xlsx: It includes the tabular data in an array format with rows the number of instances and columns. For every instance, the id is provided.
•	images: This folder contains the images, along with their ids.
•	suggested_weights: This Excel file is optional for DeepFCM-ELM. It includes the linguistic values for the input-output interconnections among DeepFCM concepts provided by experts.

Algorithm file: main.py is the DeepFCM algorithm tha integrates ELM into DeepFCM learning processes that
combines the tabular data and CNN predictions to classify the imaging case studies.

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

## Contributors

[Anna Feleki](https://emerald.uth.gr/personnel/)
[Elpiniki Papageorgiou](https://emerald.uth.gr/personnel/)
[Ioannis Apostolopoulos](https://emerald.uth.gr/personnel/)
[Nikolaos Papandrianos](https://emerald.uth.gr/personnel/)
[Serafeim Moustakidis](https://emerald.uth.gr/personnel/)

## Assets

In the /assets folder, you'll find an example file for suggested_weights, a demonstration of Grad-CAM, and a visual representation of the interconnections among concepts in a DeepFCM graph.
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
