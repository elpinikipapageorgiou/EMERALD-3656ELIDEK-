#--- IMPORT DEPENDENCIES ------------------------------------------------------
import random
from random import uniform
import numpy as np
import pandas as pd

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

class Particle:
    def __init__(self,num_clusters):
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual
        #initilization of position and velocity
        # Create the DataFrame
        num_dimensions = (num_clusters*2)+2
        df = create_dynamic_dataframe(num_clusters)
        df.to_excel('suggested.xlsx', index=False)
        arr = df.to_numpy()

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
        W2 =np.random.uniform(size = (num_dimensions,num_dimensions), low = -1, high = 1)
        np.fill_diagonal(W2, 0)
        #nullify last row of matrix
        arr[-1] = 0
        arr[-2] = 0
        
        
        self.position_i=(arr)
        #nullify last row of matrix
        W2[-1] = 0
        W2[-2] = 0
        self.velocity_i=(W2)

    # evaluate current fitness
    def evaluate(self,err_i):
        # check to see if the current position is an individual best
        if self.err_i<self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i.copy()
            self.err_best_i=err_i
 # update new particle velocity
    def update_velocity(self,pos_best_g,num_dimensions):
        
        #these values can be adjusted to the classification problem
        w=0.1       # constant inertia weight (how much to weigh the previous velocity)
        c1=0.3       # cognative constant
        c2=0.3        # social constant
        r1=uniform(0,1)
        r2=uniform(0,1)
        for i in range(0,num_dimensions):
          for j in range(0,num_dimensions):
              if(i==j):
                continue
              else:
                vel_cognitive=c1*r1*(self.pos_best_i[i][j]-self.position_i[i][j])
                vel_social=c2*r2*(pos_best_g[i][j]-self.position_i[i][j])
                self.velocity_i[i][j]=w*self.velocity_i[i][j] +vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,num_dimensions):
        for i in range(0,num_dimensions):
          for j in range(0,num_dimensions):
              if(i==j):
                continue
              else:
                self.position_i[i][j]=self.position_i[i][j]+self.velocity_i[i][j]
                # adjust maximum position if necessary
                if self.position_i[i][j]>1:
                    self.position_i[i][j]=1

                # adjust minimum position if neseccary
                if self.position_i[i][j]<-1:
                    self.position_i[i][j]=-1


#Sigmoid function                
def sig(x):
    return 1/(1 + np.exp(-x))

#Learning technique with Particle Swarm Optimization Function
def DeepFCMx(dataset, num_dimensions, num_particles, num_clusters,maxiter):
    err_best_g = float('inf')
    num_dimensions = (num_clusters*2)+2
    pos_best_g = []
    swarm = [Particle(num_clusters) for _ in range(num_particles)]
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
            particle.update_velocity(pos_best_g,num_dimensions)
            particle.update_position(num_dimensions)
        
        

    return pos_best_g, concept_evolution
