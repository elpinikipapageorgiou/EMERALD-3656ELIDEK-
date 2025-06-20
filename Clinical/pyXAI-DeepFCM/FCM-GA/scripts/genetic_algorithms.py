#Import libraries
import numpy as np

# Genetic Algorithm functions
def selection(population, fitness_scores):
    idx = np.random.choice(len(population), size=2, p=fitness_scores/np.sum(fitness_scores), replace=False)
    
    return population[idx[0]], population[idx[1]]


def crossover(crossover_rate, parent1, parent2):
    crossover_index = int(crossover_rate * parent1.shape[0])
    child1 = np.vstack((parent1[:crossover_index], parent2[crossover_index:]))
    child2 = np.vstack((parent2[:crossover_index], parent1[crossover_index:]))
    return child1, child2


def mutation(fcm, mutation_rate):
    for i in range(len(fcm)):
        for j in range(len(fcm)):
            if np.random.rand() < mutation_rate:
                mutation_step = np.random.uniform(-0.01, 0.02)  # Example: very small mutation step
                fcm[i][j] += mutation_step
    return fcm

    