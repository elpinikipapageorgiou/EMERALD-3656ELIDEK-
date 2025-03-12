#--- IMPORT DEPENDENCIES ------------------------------------------------------+
import numpy as np
from random import uniform
import pandas as pd
import random


class Particle:
    #if expert knowledge is provided in the form of linguistic values for the interconnections among input-output concepts
    def __init__(self, num_dimensions, suggested_weights=None):
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        if suggested_weights is not None:

            arr = suggested_weights.to_numpy()

            for i, row in enumerate(arr):
                for j, val in enumerate(row):
                    if arr[i][j] == "random":
                        arr[i][j]=random.uniform(-1, 1)
                    if arr[i][j]=="-VS":
                        arr[i][j]=random.uniform(-1, -0.7)
                    if arr[i][j]=='"-S"':
                        arr[i][j]=random.uniform(-0.85, -0.5)
                    if arr[i][j] =='"-M"':
                        arr[i][j]=random.uniform(-0.65, -0.35)
                    if arr[i][j] =="-W":
                        arr[i][j]=random.uniform(-0.5, -0.15)
                    if arr[i][j] =="-VW":
                        arr[i][j]=random.uniform(-0.3, 0)

                    if arr[i][j] =="VW":
                        arr[i][j]=random.uniform(0, 0.3)
                    if arr[i][j] =="W":
                        arr[i][j]=random.uniform(0.15, 0.5)
                    if arr[i][j] =="M":
                        arr[i][j]=random.uniform(0.35, 0.65)
                    if arr[i][j]=="S":
                        arr[i][j]=random.uniform(0.5, 0.85)
                    if arr[i][j]=="VS":
                        arr[i][j]=random.uniform(0.7, 1)

            np.fill_diagonal(arr, 0)

            W2 =np.random.uniform(size = (num_dimensions,num_dimensions), low = -1, high = 1)
            np.fill_diagonal(W2, 0)
            self.position_i=(arr)
            W2[-1] = 0
            self.velocity_i=(W2)
        else:

    
            self.pos_best_i=[]          # best position individual
            self.err_best_i=-1          # best error individual
            self.err_i=-1               # error individual
            #initilization of position and velocity
            W1 =np.random.uniform(size = (num_dimensions,num_dimensions), low = -1, high = 1)
            np.fill_diagonal(W1, 0)
            W2 =np.random.uniform(size = (num_dimensions,num_dimensions), low = -1, high = 1)
            np.fill_diagonal(W2, 0)
            W1[-1] = 0
            self.position_i=(W1)
            W2[-1] = 0
            self.velocity_i=(W2)

    # evaluate current fitness
    def evaluate(self,err_i):
        # check to see if the current position is an individual best
        if self.err_i<self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i.copy()
            self.err_best_i=err_i
 # update new particle velocity
    def update_velocity(self,pos_best_g, num_dimensions):
        #these values can be adjsuted according to the classification problem
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
    def update_position(self, num_dimensions):
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

#Learning technique with Fuzzy Cognitive Maps and Particle Swarm Optimization
def DeepFCM_PSO(dataset, num_dimensions, num_particles, maxiter, suggested_weights=None):
    err_best_g = float('inf')
    pos_best_g = []
    swarm = [Particle(num_dimensions=num_dimensions,suggested_weights=suggested_weights) for _ in range(num_particles)]
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
            particle.update_velocity(pos_best_g, num_dimensions)
            particle.update_position(num_dimensions)

    return pos_best_g, concept_evolution