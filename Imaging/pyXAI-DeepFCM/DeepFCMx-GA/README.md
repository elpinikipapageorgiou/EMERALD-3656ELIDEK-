# DeepFCMx-GA

# Usage

DeepFCMx-GA provides a transparent view of interconnections and their contributions to FCM's predictive accuracy.

The dataset should be initialized here:
```
dataset=pd.read_excel("dataset.xlsx", engine='openpyxl')
```

If suggested weights are provided for the initialization of interconnections among concepts, the 'suggested_weights' variable should be initialized as below
An example of an excel with initialized interconnections is provided in /assets folder
```
# excel_file_path = "suggested_weights.xlsx"
```


or else, the 'suggested_weights' variable should be equal to 'None' as it is.
```
excel_file_path=None
```
To modify the number of population and num generations, and mutation rate, adjust the following parameters accordingly
```
pop_size= 100 

#define number of generations
num_generations = 100

#define mutation rate
mutation_rate = 0.1
```
To alter the number of folds for k-fold cross validation, adjust the following parameter
```
#perform k-fold cross validation 
#Define K (num_folds) for k-fold cross validation
num_folds=10
```

To use FCM-GA, utilize the following code block:

```
best_fcm = FCM_GA(pop_size, mutation_rate, num_dimensions, training_dataset, excel_file_path)
```

# Training

The training process of FCM-GA is demonstrated.

```
def FCM_GA(population_size,  mutation_rate, num_dimensions, training_dataset, excel_file_path):
    # Generate initial population
    population = generate_population_from_excel(population_size, num_dimensions, excel_file_path)
  
    # for generation in range(num_generations):
    fitness_scores = np.array([evaluate_fitness(fcm, training_dataset, num_dimensions) for fcm in population])
  
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
    best_fcm = min(population, key=lambda fcm: evaluate_fitness(fcm, training_dataset, num_dimensions))
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
