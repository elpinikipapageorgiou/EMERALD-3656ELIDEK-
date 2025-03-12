import random
import numpy as np


def calculate_initial_weights(num_dimensions, initial_weight_matrix=None):
    if initial_weight_matrix is not None:
        initial_weight_matrix = initial_weight_matrix.to_numpy()
        for i, row in enumerate(initial_weight_matrix):
            for j, val in enumerate(row):

                if initial_weight_matrix[i][j] == "random":
                    initial_weight_matrix[i][j] = random.uniform(-1, 1)
                elif initial_weight_matrix[i][j] == "VW":  # weak
                    initial_weight_matrix[i][j] = random.uniform(0, 0.25)
                elif initial_weight_matrix[i][j] == "W":  # weak
                    initial_weight_matrix[i][j] = random.uniform(0.1, 0.4)
                elif initial_weight_matrix[i][j] == "M":  # weak
                    initial_weight_matrix[i][j] = random.uniform(0.35, 0.65)
                elif initial_weight_matrix[i][j] == "S":  # strong
                    initial_weight_matrix[i][j] = random.uniform(0.55, 0.85)
                elif initial_weight_matrix[i][j] == "VS":  # very strong
                    initial_weight_matrix[i][j] = random.uniform(0.75, 1)

                elif initial_weight_matrix[i][j] == '"-VW"':  # Minus Very Weak
                    initial_weight_matrix[i][j] = random.uniform(-0.25, 0)
                elif initial_weight_matrix[i][j] == '"-W"':  # Minus Weak
                    initial_weight_matrix[i][j] = random.uniform(-0.4, -0.1)
                elif initial_weight_matrix[i][j] == '"-M"':  # weak
                    initial_weight_matrix[i][j] = random.uniform(-0.65, -0.35)
                elif initial_weight_matrix[i][j] == '"-S"':  # Strong
                    initial_weight_matrix[i][j] = random.uniform(-0.85, -0.55)
                elif initial_weight_matrix[i][j] == '"-VS"':  # Minus Very Strong
                    initial_weight_matrix[i][j] = random.uniform(-1, -0.75)
        

    else:
        
        initial_weight_matrix=np.random.uniform(-1, 1, (num_dimensions, num_dimensions))
                    
    
    np.fill_diagonal(initial_weight_matrix, 0) 
    return initial_weight_matrix

