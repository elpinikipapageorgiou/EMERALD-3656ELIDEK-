#--- IMPORT DEPENDENCIES ------------------------------------------------------+
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def create_feature_maps(addrs, num_clusters,pixel_size,feature_layer_model):
    class1_Feature_maps=[]
    class0_Feature_maps=[]
    initial_training_featuresss = []
    full_actual_output=[]
    # print(addrs)
    rgb_training_features=[]
    for img in addrs:  
        if "class0" in str(img):
            full_actual_output.append(0)
        if "class1" in str(img):
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
  