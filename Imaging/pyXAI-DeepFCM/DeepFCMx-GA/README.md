# DeepFCMx-GA

DeepFCMx is an FCM-based classification framework designed for medical image analysis, integrating CNNs with FCMs. The methodology begins with CNN training, where the feature maps are extracted from the last convolutional layer, capturing critical spatial and textural patterns within the medical images. These feature maps represent high-dimensional image descriptors, which serve as the primary input for the subsequent clustering stage. The feature maps are grouped based on the class labels of the corresponding images, and they are clustered, with the number of clusters optimized through experimentation to achieve optimal classification performance. To establish structured representations of image data, Euclidean distance is employed to compute similarities between extracted feature maps and the cluster centroids. This transformation allows the generation of a structured dataset, which serves as the input to the DeepFCMx classifier. In DeepFCMx-GA, GA is utilized to calculate the interconnections among concepts.

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
# Set number of population size, generations, crossover rate, mutation rate, and number of iterations of DeepFCMx-GA
pop_size= 100 

num_generations = 100

crossover_rate = 0.5

mutation_rate = 0.1

num_iterations=25

# Usage

To use DeepFCMx-GA, utilize the following code block:

```
best_position = DeepFCMx_GA(pop_size, mutation_rate, num_dimensions, training_dataset, num_clusters)
```

# Training

The training process of DeepFCMx-GA is demonstrated.

```
def DeepFCMx_GA(population_size,  mutation_rate, num_dimensions, dataset, num_clusters):
    # Generate initial population
    population = generate_population_from_excel(population_size, num_dimensions, num_clusters)
    
    # for generation in range(num_generations):
    fitness_scores = np.array([evaluate_fitness(fcm, dataset, num_dimensions) for fcm in population])
    
    # Selection
    parents = [selection(population, fitness_scores) for _ in range(population_size//2)]
    
    # Crossover
    children = [crossover(parent1, parent2) for parent1, parent2 in parents]
    children = [child for sublist in children for child in sublist]

    # Mutation
    mutated_children = [mutation(child, mutation_rate) for child in children]

    # Create new population
    population = mutated_children
    
    # Optionally, you can track the best FCM in each generation
    best_fcm = min(population, key=lambda fcm: evaluate_fitness(fcm, dataset, num_dimensions))
    # print(f"Generation {generation+1}: Best Fitness Score: {evaluate_fitness(best_fcm, dataset)}")
    
    return best_fcm
```

# Interpretation

This block of code will demonstrate the interconnections among concepts

```
plot_FCM_weight_matrix_graph(column_names, last_column_except_last)
```

![Plot](assets/FCM-GA_CAD_Clinical.jpg)

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
