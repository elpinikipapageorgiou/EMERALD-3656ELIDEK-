
from genetic_functions import generate_population_from_excel, selection, crossover, mutation
import numpy as np
from evaluate_fitness_function import evaluate_fitness, sig


#--- IMPORT DEPENDENCIES ------------------------------------------------------+
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def DeepFCMx_GA(population_size, crossover_rate, mutation_rate, num_dimensions, dataset, num_clusters):
    # Generate initial population
    population = generate_population_from_excel(population_size, num_dimensions, num_clusters)
    
    # for generation in range(num_generations):
    fitness_scores = np.array([evaluate_fitness(fcm, dataset, num_dimensions) for fcm in population])
    
    # Selection
    parents = [selection(population, fitness_scores) for _ in range(population_size//2)]
    
    # Crossover
    children = [crossover(crossover_rate, parent1, parent2) for parent1, parent2 in parents]
    children = [child for sublist in children for child in sublist]

    # Mutation
    mutated_children = [mutation(child, mutation_rate) for child in children]

    # Create new population
    population = mutated_children
    
    # Optionally, you can track the best FCM in each generation
    best_fcm = min(population, key=lambda fcm: evaluate_fitness(fcm, dataset, num_dimensions))
    # print(f"Generation {generation+1}: Best Fitness Score: {evaluate_fitness(best_fcm, dataset)}")
    
    return best_fcm

def predict_fcm(fcm, data):
    predicted_results = np.zeros((len(data), fcm.shape[0]))
    for index, testing_row in enumerate(data):
        for i in range(fcm.shape[0]):
            sum_temp = np.sum([fcm[j][i] * testing_row[j] for j in range(fcm.shape[0]) if i != j])
            predicted_results[index][i] = sig(sum_temp)

    testing_last_element_fcm_output = predicted_results[:, -1]
    return testing_last_element_fcm_output


def create_feature_maps(addrs, num_clusters,pixel_size,feature_layer_model, class0_name, class1_name):
    class1_Feature_maps=[]
    class0_Feature_maps=[]
    initial_training_featuresss = []
    full_actual_output=[]
    # print(addrs)
    rgb_training_features=[]
    for img in addrs:  
        if class0_name in str(img):
            full_actual_output.append(0)
        if class1_name in str(img):
            full_actual_output.append(1)
            
            
        img1 = load_img(img, target_size=(pixel_size, pixel_size, 3))
        x = img_to_array(img1)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0


        features = feature_layer_model.predict(x)

    

        rgb_training_features.append(features)

    initial_training_featuresss = np.array(rgb_training_features)

    #Save the extract feature maps
    np.save('data_cnn.npy', initial_training_featuresss)
    initial_training_featuresss = np.load('data_cnn.npy')

    #divide feature maps according to their belonging class
    for feature_index, y_index in zip(initial_training_featuresss, full_actual_output):
        if y_index==1:
            class1_Feature_maps.append(feature_index)
        if y_index==0:
            class0_Feature_maps.append(feature_index)

    class1_Feature_maps = np.array(class1_Feature_maps)
    class0_Feature_maps = np.array(class0_Feature_maps)

    # Normalize feature vectors
    class1_Feature_maps = (class1_Feature_maps - np.min(class1_Feature_maps)) / (np.max(class1_Feature_maps) - np.min(class1_Feature_maps))
    class0_Feature_maps = (class0_Feature_maps - np.min(class0_Feature_maps)) / (np.max(class0_Feature_maps) - np.min(class0_Feature_maps))
    dataset = create_centroids(num_clusters, class1_Feature_maps,class0_Feature_maps, initial_training_featuresss, full_actual_output)
    return dataset

def create_centroids(num_clusters, class1_Feature_maps,class0_Feature_maps,initial_training_featuresss, full_actual_output ):
    #---------------------Class 0-------------------#
    class0_Feature_maps = np.array(class0_Feature_maps)
    class0_Feature_maps = class0_Feature_maps.reshape((-1, class0_Feature_maps.shape[-1]))


    kmeans_class0 = KMeans(n_clusters=num_clusters, n_init='auto')
    kmeans_class0.fit(class0_Feature_maps)
    centroids_class0= kmeans_class0.cluster_centers_

    # print("centroids_class0", centroids_class0)
    list_centroids_class0=[]
    for i in range(num_clusters):
        list_centroids_class0.append(centroids_class0[i])
    
    #---------------------Class1-------------------#
    print("----centroids-----")
    class1_Feature_maps = np.array(class1_Feature_maps)
    class1_Feature_maps = class1_Feature_maps.reshape((-1, class1_Feature_maps.shape[-1]))


    kmeans_class1 = KMeans(n_clusters=num_clusters, n_init='auto')
    kmeans_class1.fit(class1_Feature_maps)
    centroids_class1= kmeans_class1.cluster_centers_


    #append the centroid of each cluster
    list_centroids_class1=[]
    for i in range(num_clusters):
        list_centroids_class1.append(centroids_class1[i])


    centroids =[]
    centroids.append(list_centroids_class0)
    centroids.append(list_centroids_class1)
    centroids=np.array(centroids)

    # compute the sum of all centroids of both two classes
    sum_centroids = np.sum(centroids)
    dataset= create_similarities(initial_training_featuresss, num_clusters, centroids_class1, centroids_class0,sum_centroids, full_actual_output )
    return dataset

def create_similarities(initial_training_featuresss, num_clusters, centroids_class1, centroids_class0,sum_centroids, full_actual_output ):
    #compute the similarities of centroids for all feature vectors
    rows=[]
    similarities=[]
    for training_row in initial_training_featuresss:
        rows = []
        # Loop for centroids_class0
        for i in range(num_clusters):
            rows.append(1 - (np.linalg.norm(training_row - centroids_class0[i]) / sum_centroids))
            
        # Loop for centroids_class1
        for i in range(num_clusters):
            rows.append(1 - (np.linalg.norm(training_row - centroids_class1[i]) / sum_centroids))
        
        
        similarities.append(rows)

    similarities = np.array(similarities)

    similarities =  (similarities - np.min(similarities)) / np.ptp(similarities)


    inverted_list = [1 if item == 0 else 0 for item in full_actual_output]

    #construct the dataset, which includes the similarities along with the output
    inverted_list = np.array(inverted_list)
    full_actual_output=np.array(full_actual_output)

    similarity_columns = [similarities[:, i].reshape(-1, 1) for i in range(num_clusters * 2)]

    # Append the additional columns
    similarity_columns.extend([inverted_list.reshape(-1, 1), full_actual_output.reshape(-1, 1)])

    # Concatenate all columns along the second axis
    result = np.concatenate(similarity_columns, axis=1)

    print("shape")
    print(result.shape)

    df = pd.DataFrame(result)

    # save the dataframe to an Excel file
    df.to_excel('full_dataset.xlsx', index=False)

    return df
  
