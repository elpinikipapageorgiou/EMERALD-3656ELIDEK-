# DeepFCMx-PSO

DeepFCMx is an FCM-based classification framework designed for medical image analysis, integrating CNNs with FCMs. The methodology begins with CNN training, 
where the feature maps are extracted from the last convolutional layer, capturing critical spatial and textural patterns within the medical images. These feature maps represent high-dimensional image descriptors, which serve as the primary input for the subsequent clustering stage. The feature maps are grouped based on the class labels
of the corresponding images, and they are clustered, with the number of clusters optimized through experimentation to achieve optimal classification performance.
To establish structured representations of image data, Euclidean distance is employed to compute similarities between extracted feature maps and the cluster centroids. 
This transformation allows the generation of a structured dataset, which serves as the input to the DeepFCMx classifier. 
In DeepFCMx-PSO, PSO is utilized to calculate the interconnections among concepts.

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

# Set number of epochs, particles, and fold for DeepFCMx-PSO

```
epoch = 25
num_particles = 40
folds=10
```

# Usage

DeepFCMx provides a transparent view of interconnections and their contributions to FCM's predictive accuracy, with solely imaging data as input.

To use DeepFCMx, utilize the following code block:

```
best_position, concept_evolution  = DeepFCMx(training_dataset,  num_dimensions=num_dimensions, num_particles=num_particles, maxiter=epoch)
```

# Training

The training process of DeepFCMx-PSO is demonstrated.

```
#Learning technique with Particle Swarm Optimization Function
def DeepFCMx(dataset, num_dimensions, num_particles, maxiter):
    err_best_g = float('inf')
    pos_best_g = []
    swarm = [Particle() for _ in range(num_particles)]
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
            particle.update_velocity(pos_best_g)
            particle.update_position()
    
    

    return pos_best_g, concept_evolution

```

## Dataset Description:

•	all_images: It includes the imaging dataset

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
