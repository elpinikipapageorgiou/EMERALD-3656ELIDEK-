import numpy as np
import pandas as pd
import random

# Genetic Algorithm functions
def selection(population, fitness_scores):
    # normalized_fitness = (fitness_scores - np.min(fitness_scores)) / (np.max(fitness_scores) - np.min(fitness_scores))
    idx = np.random.choice(len(population), size=2, p=fitness_scores/np.sum(fitness_scores), replace=False)
    
    return population[idx[0]], population[idx[1]]


def crossover(parent1, parent2):
    crossover_index = int(crossover_rate * parent1.shape[0])
    child1 = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.vstack((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2


def mutation(fcm, mutation_rate):
    for i in range(len(fcm)):
        for j in range(len(fcm)):
            if np.random.rand() < mutation_rate:
                mutation_step = np.random.uniform(-0.01, 0.02)  # Example: very small mutation step
                fcm[i][j] += mutation_step
    return fcm

def create_dynamic_dataframe(num_clusters):
    # Initialize column names for clusters of each class
    columns = [f'c{i+1}' for i in range(num_clusters * 2)] + ['Class0', 'Class1']
    
    # Initialize rows based on the number of clusters
    num_rows_per_class = num_clusters
    total_rows = num_rows_per_class * 2
    data = []
    
    # Create rows for the first class (positive)
    for _ in range(num_rows_per_class):
        row = ['positive'] * num_clusters + ['negative'] * num_clusters + ['negative_output', 'positive_output']
        data.append(row)
    
    # Create rows for the second class (negative)
    for _ in range(num_rows_per_class):
        row = ['negative'] * num_clusters + ['positive'] * num_clusters + ['positive_output', 'negative_output']
        data.append(row)
    
    # Add rows with zeros at the end
    for _ in range(2):
        row = [0] * (num_clusters * 2) + [0, 0]
        data.append(row)

    # Create the DataFrame
    df = pd.DataFrame(data, columns=columns)
    return df



def generate_population_from_excel(population_size, num_dimensions, num_clusters):
    population = []

    # Read the suggested weights from the Excel file
    num_dimensions = (num_clusters*2)+2
    df = create_dynamic_dataframe(num_clusters)
    df.to_excel('suggested.xlsx', index=False)

    arr = df.to_numpy()

    for _ in range(population_size):
        for i in range(0,num_dimensions):
            for j in range(0,num_dimensions):

                if arr[i][j]=="negative":
                    arr[i][j]=random.uniform(-1, -0.5)
                if arr[i][j]=="positive":
                    arr[i][j]=random.uniform(0.5, 1)

                if arr[i][j]=="positive_output":
                    arr[i][j]=random.uniform(0, 1)
                if arr[i][j]=="negative_output":
                    arr[i][j]=random.uniform(-1, 0)

        np.fill_diagonal(arr, 0)
        arr[-1] = 0
        arr[-2] = 0
        population.append(arr)

    return population
