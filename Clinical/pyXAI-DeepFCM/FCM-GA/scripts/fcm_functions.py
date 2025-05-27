#Import libraries
import numpy as np
import pandas as pd

#Import function
from generate_population import generate_population_from_excel
from genetic_algorithms import selection, crossover, mutation

#Sigmoid function
def sig(x):
    return 1/(1 + np.exp(-x))

def evaluate_fitness(fcm, data, num_dimensions):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    dataset = data.values
    fitness_result = 0

    for row in dataset:
        fcm_output = np.zeros(num_dimensions)
        for i in range(num_dimensions):
            sum_temp = sum(fcm[j][i] * row[j] for j in range(num_dimensions) if i != j)
            fcm_output[i] = sig(sum_temp)

        fitness_result += np.square(fcm_output[-1] - row[-1])
    
    return fitness_result

def FCM_GA(population_size, crossover_rate,  mutation_rate, num_dimensions, training_dataset, excel_file_path):
    # Generate initial population
    population = generate_population_from_excel(population_size, num_dimensions, excel_file_path)
    
    fitness_scores = np.array([evaluate_fitness(fcm, training_dataset, num_dimensions) for fcm in population])
    
    # Selection
    parents = [selection(population, fitness_scores) for _ in range(population_size//2)]
    
    # Crossover
    children = [crossover(crossover_rate, parent1, parent2) for parent1, parent2 in parents]
    children = [child for sublist in children for child in sublist]

    # Mutation
    mutated_children = [mutation(child, mutation_rate) for child in children]

    # Create new population
    population = mutated_children

    best_fcm = min(population, key=lambda fcm: evaluate_fitness(fcm, training_dataset, num_dimensions))
    
    return best_fcm

def predict_fcm(fcm, data):
    predicted_results = np.zeros((len(data), fcm.shape[0]))
    for index, testing_row in enumerate(data):
        for i in range(fcm.shape[0]):
            sum_temp = np.sum([fcm[j][i] * testing_row[j] for j in range(fcm.shape[0]) if i != j])
            predicted_results[index][i] = sig(sum_temp)

    testing_last_element_fcm_output = predicted_results[:, -1]
    return testing_last_element_fcm_output
