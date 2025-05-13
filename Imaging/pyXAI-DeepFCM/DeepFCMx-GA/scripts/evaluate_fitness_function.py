import numpy as np
import pandas as pd
import math

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

#calculate deviation for the performance metrics
def calculate_deviation(values):
    # Step 1
    mean = sum(values) / len(values)

    # Step 2
    differences = [value - mean for value in values]

    # Step 3
    squared_differences = [diff ** 2 for diff in differences]

    # Step 4
    mean_squared_differences = sum(squared_differences) / len(squared_differences)

    # Step 5
    standard_deviation = math.sqrt(mean_squared_differences)

    return standard_deviation