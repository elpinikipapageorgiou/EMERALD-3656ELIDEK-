# DeepFCM-PSO

This is the official implementation of

[https://doi.org/10.3390/app132111953](https://doi.org/10.3390/app132111953),

[https://doi.org/10.1007/978-3-031-39965-7_2](https://doi.org/10.1007/978-3-031-39965-7_2),

[https://ieeexplore.ieee.org/abstract/document/10345912](https://ieeexplore.ieee.org/abstract/document/10345912)

[https://ieeexplore.ieee.org/abstract/document/10786612](https://ieeexplore.ieee.org/abstract/document/10786612)

# Paper Abstract

[ https://doi.org/10.3390/app132111953](),

Myocardial Perfusion Imaging (MPI) has played a central role in the non-invasive identification of patients with Coronary Artery Disease (CAD). Clinical factors, such as recurrent diseases, predisposing factors, and diagnostic tests, also play a vital role. However, none of these factors offer a straightforward and reliable indication, making the diagnosis of CAD a non-trivial task for nuclear medicine experts. While Machine Learning (ML) and Deep Learning (DL) techniques have shown promise in this domain, their “black-box” nature remains a significant barrier to clinical adoption, a challenge that the existing literature has not yet fully addressed. This study introduces the Deep Fuzzy Cognitive Map (DeepFCM), a novel, transparent, and explainable model designed to diagnose CAD using imaging and clinical data. DeepFCM employs an inner Convolutional Neural Network (CNN) to classify MPI polar map images. The CNN’s prediction is combined with clinical data by the FCM-based classifier to reach an outcome regarding the presence of CAD. For the initialization of interconnections among DeepFCM concepts, expert knowledge is provided. Particle Swarm Optimization (PSO) is utilized to adjust the weight values to the correlated dataset and expert knowledge. The model’s key advantage lies in its explainability, provided through three main functionalities. First, DeepFCM integrates a Gradient Class Activation Mapping (Grad-CAM) algorithm to highlight significant regions on the polar maps. Second, DeepFCM discloses its internal weights and their impact on the diagnostic outcome. Third, the model employs the Generative Pre-trained Transformer (GPT) version 3.5 model to generate meaningful explanations for medical staff. Our dataset comprises 594 patients, who underwent invasive coronary angiography (ICA) at the department of Nuclear Medicine of the University Hospital of Patras in Greece. As far as the classification results are concerned, DeepFCM achieved an accuracy of 83.07%, a sensitivity of 86.21%, and a specificity of 79.99%. The explainability-enhancing methods were assessed by the medical experts on the authors’ team and are presented within. The proposed framework can have immediate application in daily routines and can also serve educational purposes.
Keywords:  fuzzy cognitive maps; particle swarm optimization; convolutional neural networks; classification; feature selection; Grad-CAM; coronary artery disease; natural language processing

# Multimodal approach (Clinical + Imaging Data)

# Define class labels

class0_name='normal'

class1_name='pathological'

# To set the path of the imaging folder

Specify the input size (in pixels) for resizing all images:

pixel_size = 300

Ensure that your images have one of the following extensions

```
def read_and_process_image(list_of_images):
    X = []
    for img in list_of_images:
        image = cv2.imread(img)
        X.append(cv2.resize(image, (pixel_size, pixel_size), interpolation=cv2.INTER_CUBIC))

    # Define the output of each image to a different list
    y = [0 if class0_name in addr else 1 for addr in list_of_images]

    return X, y
```

Set the number of epochs, batch size, and architecture of RGB-CNN.

epochs = 200

batch_size = 32

# CNN Architecture (RGB-CNN)

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

# Clinical Data Normalization

Select which columns to be normalized with the Min-max normalization

columns_to_normalize = ['column1', 'column2']  # Add any other columns you need to normalize

# Define hyper-parameters for DeepFCM-PSO

##Define number of particles

num_particles = 40

#define number of epochs

epoch = 20

#Perform k-fold cross validation

folds=10

# Expert knowledge (if provided)

If expert knowledge is provided in the form of fuzzy sets, the user must define the Excel file.

suggested_weights = pd.read_excel("suggested_weights.xlsx", engine='openpyxl')

# Usage

This multimodal approach enhances interpretability by integrating both clinical and imaging data, providing a transparent view of interconnections and their contributions to DeepFCM's predictive accuracy.

To use DeepFCM-PSO, utilize the following code block:

```
best_position,concept_evolution = DeepFCM_PSO(training_dataset,
                                        num_dimensions=num_dimensions,
                                        num_particles=num_particles, maxiter=epoch,
                                        suggested_weights=suggested_weights)

```

# Training

The training process of DeepFCM-PSO is demonstrated.

```
def DeepFCM-PSO(dataset, num_dimensions, num_particles, maxiter, suggested_weights=None):
    err_best_g = float('inf')
    pos_best_g = []
    swarm = [Particle(num_dimensions=num_dimensions,suggested_weights=suggested_weights) for _ in range(num_particles)]
    concept_evolution = np.zeros((maxiter, num_dimensions))  # Record concept values over epochs

    for k in range(maxiter):
        # Evaluate fitness of each particle at its current position
        for particle in swarm:
            fitness_result = 0
            for row in dataset:
                fcm_output = [0] * num_dimensions
                for i in range(num_dimensions):
                    sum_temp = sum(particle.position_i[j][i] * row[j] for j in range(num_dimensions) if i != j)
                    fcm_output[i] = sig(sum_temp)

                fitness_result += np.square(fcm_output[-1] - row[-1])

            particle.err_i = fitness_result / len(dataset)
            particle.evaluate(particle.err_i)

            # Update the global best position
            if particle.err_i < err_best_g or k == 0:
                pos_best_g = particle.position_i.copy()
                err_best_g = particle.err_i

        # Record the concept evolution
        concept_evolution[k, :] = [sig(sum(pos_best_g[j][i] * value for j, value in enumerate(row))) for i in range(num_dimensions)]

        # Update velocity and position of each particle after fitness evaluation
        for particle in swarm:
            particle.update_velocity(pos_best_g, num_dimensions)
            particle.update_position(num_dimensions)

    return pos_best_g, concept_evolution

```

# Interpretation

This block of code will demonstrate the interconnections among concepts

```
plot_FCM_weight_matrix_graph(column_names, last_column_except_last)
```

![Plot](assets/DeepFCM_PSO_CAD.jpg)

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

•	dataset.xlsx: It includes the clinical data in an array format with rows the number of instances and columns. For every instance, the id is provided.
•	images: This folder contains the images, along with their ids.
•	suggested_weights: This Excel file is optional for DeepFCM-PSO. It includes the linguistic values for the input-output interconnections among DeepFCM concepts provided by experts.

Algorithm file: main.py is the DeepFCM algorithm tha integrates PSO into DeepFCM learning processes that
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

## Supervisor

[Elpiniki Papageorgiou](https://emerald.uth.gr/personnel/)

## Contributors

[Anna Feleki](https://emerald.uth.gr/personnel/)
[Elpiniki Papageorgiou](https://emerald.uth.gr/personnel/)
[Ioannis Apostolopoulos](https://emerald.uth.gr/personnel/)
[Nikolaos Papandrianos](https://emerald.uth.gr/personnel/)
[Serafeim Moustakidis](https://emerald.uth.gr/personnel/)

## Citation

If you find this work useful, please cite our paper:

Anna Feleki, Ioannis Apostolopoulos, Serafeim Moustakidis, Elpiniki Papageorgiou, Nikolaos Papathanasiou, Dimitrios Apostolopoulos, Nikolaos Papandrianos, Explainable Deep Fuzzy Cognitive Map Diagnosis of Coronary Artery Disease: Integrating Myocardial Perfusion Imaging, Clinical Data, and Natural Language Insights. *Appl. Sci.*  **2023** ,  *13* , 11953. https://doi.org/10.3390/app132111953

Anna Feleki, Ioannis Apostolopoulos, Konstantinos Papageorgiou, Elpiniki Papageorgiou, Dimitrios Apostolopoulos, Nikolaos Papandrianos, N.I. (2023). A Fuzzy Cognitive Map Learning Approach for Coronary Artery Disease Diagnosis in Nuclear Medicine. In: Massanet, S., Montes, S., Ruiz-Aguilera, D., González-Hidalgo, M. (eds) Fuzzy Logic and Technology, and Aggregation Operators. EUSFLAT AGOP 2023 2023. https://doi.org/10.1007/978-3-031-39965-7_2

Anna Feleki, Ioannis Apostolopoulos, Elpiniki Papageorgiou, Serafeim Moustakidis, Nikolaos D. Papathanasiou, Dimitrios J. Apostolopoulos ., "Deep Fuzzy Cognitive Map methodology for Non-Small Cell Lung Cancer diagnosis based on Positron Emission Tomography imaging,"  *2023 14th International Conference on Information, Intelligence, Systems & Applications (IISA)* , Volos, Greece, 2023, pp. 1-6, doi: 10.1109/IISA59645.2023.10345912.

## Assets

In the /assets folder, you'll find an example file for suggested_weights, and a visual representation of the interconnections among concepts in a DeepFCM graph.
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
