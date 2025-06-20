# RGB-CNN

This is the official implementation of:

[https://dl.acm.org/doi/abs/10.1145/3503823.3503911](https://dl.acm.org/doi/abs/10.1145/3503823.3503911)
[https://www.mdpi.com/2077-0383/11/13/3918](https://www.mdpi.com/2077-0383/11/13/3918)
[https://ieeexplore.ieee.org/abstract/document/9904340](https://ieeexplore.ieee.org/abstract/document/9904340)
[https://link.springer.com/article/10.1007/s12149-022-01762-4](https://link.springer.com/article/10.1007/s12149-022-01762-4)
[https://www.mdpi.com/2076-3417/12/15/7592](https://www.mdpi.com/2076-3417/12/15/7592)

# Paper Abstract

[https://www.mdpi.com/2076-3417/12/15/7592](https://www.mdpi.com/2076-3417/12/15/7592)
Background: This study targets the development of an explainable deep learning methodology for the automatic classification of coronary artery disease, utilizing SPECT MPI images. Deep learning is currently judged as non-transparent due to the model’s complex non-linear structure, and thus, it is considered a «black box», making it hard to gain a comprehensive understanding of its internal processes and explain its behavior. Existing explainable artificial intelligence tools can provide insights into the internal functionality of deep learning and especially of convolutional neural networks, allowing transparency and interpretation. Methods: This study seeks to address the identification of patients’ CAD status (infarction, ischemia or normal) by developing an explainable deep learning pipeline in the form of a handcrafted convolutional neural network. The proposed RGB-CNN model utilizes various pre- and post-processing tools and deploys a state-of-the-art explainability tool to produce more interpretable predictions in decision making. The dataset includes cases from 625 patients as stress and rest representations, comprising 127 infarction, 241 ischemic, and 257 normal cases previously classified by a doctor. The imaging dataset was split into 20% for testing and 80% for training, of which 15% was further used for validation purposes. Data augmentation was employed to increase generalization. The efficacy of the well-known Grad-CAM-based color visualization approach was also evaluated in this research to provide predictions with interpretability in the detection of infarction and ischemia in SPECT MPI images, counterbalancing any lack of rationale in the results extracted by the CNNs. Results: The proposed model achieved 93.3% accuracy and 94.58% AUC, demonstrating efficient performance and stability. Grad-CAM has shown to be a valuable tool for explaining CNN-based judgments in SPECT MPI images, allowing nuclear physicians to make fast and confident judgments by using the visual explanations offered. Conclusions: Prediction results indicate a robust and efficient model based on the deep learning methodology which is proposed for CAD diagnosis in nuclear medicine.
Keywords: deep learning; convolutional neural network; explainable artificial intelligence; Grad-CAM

# Usage

RGB-CNN is a custom-built Convolutional Neural Network (CNN), specifically tailored to each classification task by adapting its architecture accordingly. Its flexibility and effectiveness have been demonstrated across multiple case studies.

To use RGB-CNN, pass the image directory to the data_dir parameter:

```
data_dir = 'all_images/'
```
Change the name of your output classes 
```
class0_name='class0'
class1_name='class1'
```

# Set-up hyperparameters
```
# Batch size (should be a factor of 2.***4,8,16,32,64...***)
batch_size = 16

# Dropout rate
drop_rate = 0.1
 
# Number of epochs
num_epochs = 300
```
# Usage
To modify the RGB-CNN architecture, please add/remove the nodes and layers in the following block code:

```
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
```

# Training

The training process of RGB-CNN is demonstrated.

```
history = model.fit(train_generator,
                epochs=num_epochs,
                validation_data = val_generator,
                validation_steps = ntest // batch_size,
                callbacks=[early_stopping])
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

•	all_images: It includes the all the available images.

Algorithm file: main.py is the RGB-CNN algorithm.

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
-keras
-tensorflow

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

Papandrianos, N.I., Apostolopoulos, I.D., Feleki, A. et al. Deep learning exploration for SPECT MPI polar map images classification in coronary artery disease. Ann Nucl Med 36, 823–833 (2022). https://doi.org/10.1007/s12149-022-01762-4

Papandrianos, N.I.; Feleki, A.; Moustakidis, S.; Papageorgiou, E.I.; Apostolopoulos, I.D.; Apostolopoulos, D.J. An Explainable Classification Method of SPECT Myocardial Perfusion Images in Nuclear Cardiology Using Deep Learning and Grad-CAM. Appl. Sci. 2022, 12, 7592. https://doi.org/10.3390/app12157592

N. I. Papandrianos, A. Feleki, S. Moustakidis and E. I. Papageorgiou, "A Convolutional Neural Network-based explainable classification method of SPECT myocardial perfusion images in nuclear cardiology," 2022 13th International Conference on Information, Intelligence, Systems & Applications (IISA), Corfu, Greece, 2022, pp. 1-7, doi: 10.1109/IISA56318.2022.9904340.

Papandrianos, N.I.; Feleki, A.; Papageorgiou, E.I.; Martini, C. Deep Learning-Based Automated Diagnosis for Coronary Artery Disease Using SPECT-MPI Images. J. Clin. Med. 2022, 11, 3918. https://doi.org/10.3390/jcm11133918

Papandrianos, N.I.; Feleki, A.; Papageorgiou, E.I.; Martini, C. Deep Learning-Based Automated Diagnosis for Coronary Artery Disease Using SPECT-MPI Images. J. Clin. Med. 2022, 11, 3918. https://doi.org/10.3390/jcm11133918

