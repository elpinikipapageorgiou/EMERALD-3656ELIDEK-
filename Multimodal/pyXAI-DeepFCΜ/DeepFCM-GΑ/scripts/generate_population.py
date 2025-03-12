#Import libraries
import numpy as np
import pandas as pd
import random

def generate_population_from_excel(population_size, num_concepts, excel_file_path=None):
    population = []
    #if expert knowledge is provided
    # Read the suggested weights from the Excel file
    if excel_file_path !=None:
        df = pd.read_excel(excel_file_path, nrows=num_concepts, engine='openpyxl')
        arr = df.to_numpy()

        for _ in range(population_size):
            fcm = np.zeros((num_concepts, num_concepts))

            for i in range(num_concepts):
                for j in range(num_concepts):
                    if arr[i][j] == "random":
                        fcm[i][j] = random.uniform(-1, 1)
                    elif arr[i][j] == "VW":  # weak
                        fcm[i][j] = random.uniform(0, 0.25)
                    elif arr[i][j] == "W":  # weak
                        fcm[i][j] = random.uniform(0.1, 0.4)
                    elif arr[i][j] == "M":  # weak
                        fcm[i][j] = random.uniform(0.35, 0.65)
                    elif arr[i][j] == "S":  # strong
                        fcm[i][j] = random.uniform(0.55, 0.85)
                    elif arr[i][j] == "VS":  # very strong
                        fcm[i][j] = random.uniform(0.75, 1)

                    elif arr[i][j] == '"-VW"':  # Minus Very Weak
                        fcm[i][j] = random.uniform(-0.25, 0)
                    elif arr[i][j] == '"-W"':  # Minus Weak
                        fcm[i][j] = random.uniform(-0.4, -0.1)
                    elif arr[i][j] == '"-M"':  # Minus Medium
                        fcm[i][j] = random.uniform(-0.65, -0.35)
                    elif arr[i][j] == '"-S"':  # Minus Strong
                        fcm[i][j] = random.uniform(-0.85, -0.55)
                    elif arr[i][j] == '"-VS"':  # Minus Very Strong
                        fcm[i][j] = random.uniform(-1, -0.75)

            np.fill_diagonal(fcm, 0)
            fcm[-1] = 0
            population.append(fcm)
    
    else:
    #if no expert knowledge is provided
        for _ in range(population_size):
            fcm=np.random.uniform(size = (num_concepts,num_concepts), low = -1, high = 1)
            fcm[-1] = 0
            population.append(fcm)
        

    return population
