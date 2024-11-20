# DeepFCMx

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

## Contributors

[Anna Feleki](https://emerald.uth.gr/personnel/)
[Elpiniki Papageorgiou](https://emerald.uth.gr/personnel/)
[Ioannis Apostolopoulos](https://emerald.uth.gr/personnel/)
[Nikolaos Papandrianos](https://emerald.uth.gr/personnel/)
[Serafeim Moustakidis](https://emerald.uth.gr/personnel/)
