#A novel method for FCM learning with the employment of Neural Networks. 
#Created by Theotziol on 15/9/2023
#Contact info ttziolas@uth.gr


import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import pandas as pd
#from sklearn.preprocessing import MinMaxScaler

import time
# from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


### Preprocessing functions
def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())

    return df_norm

def convert_to_categorical(df, column, shuffle = True):
    '''
    converts a pandas dataframe output column into categorical columns (1 column per class label)
    use the shuffle boolean to randomly re-arrange the rows
    '''
    df = df.copy()
    labels = np.unique(df[column])
    zeros = np.zeros(len(df))
    for i in range(len(labels)):
        df[labels[i]] = zeros
        df[labels[i]][df[column] == labels[i]] = 1
    df.pop(column)
    if shuffle:
        df = df.sample(frac = 1).reset_index(drop=True)
    return df

def split_labels(df, labels_index = -1, value = 0.5, shuffle = True):
    '''
    function for classification tasks. 
    It splits the original dataframe into 1) input_df and 2) labels_df. 
    FCM classification requires the input vector to have the shape of inputs+outputs.
    Thus, this function separates the original label values and replaces them with the dummy {value}.
    Args:
        df: the pandas dataframe
        labels_index : int. the column which the separation will began. (default = -1)
        value : float. the dummy value that will be given to the input vector for the labels (default = 0.5)
        shuffle : Boolean. whether to shuffle df (Default = True)  
    Returns:
        input_df : pandas dataframe
        df_labels : pandas dataframe
    '''
    if shuffle:
        input_df = df.copy().sample(frac=1).reset_index(drop=True)
    else:
        input_df = df.copy()
    df_labels = input_df[df.columns[labels_index:]]
    input_df[df.columns[labels_index:]] = value
    return input_df, df_labels


###sigmoid tensorflow
def sigmoid(x,l = 1):
    return 1/(1 + tf.math.exp(-x *l))


###load model and dataset
model = keras.models.load_model("models\\cad2_experts_2lambda_1iters.keras", compile = False) # This model requires l=2 in sigmoid and 1 iterations in FCM inference.
cad2 = pd.read_csv("data\\new_cad_with_experts\\cad_full_with_artificial_samples.csv", delimiter = ';', decimal = ',') #I used csv files in my experiments
cad2.dropna(inplace = True)
cad2_categorical = convert_to_categorical(cad2, cad2.columns[-1])

#I use the min-max normalization. The values are expected in [0,1]. **IMPORTANT** 1 is the maximum of the WHOLE COLUMN and 0 the minimum 
cad2_categorical = min_max_scaling(cad2_categorical) 

# Here i copy the actual labels and I keep the class columns (2 last columns for binary classification) with dummy values (0.5)
cad2_input, cad2_labels = split_labels(cad2_categorical, -2)

#Testing
example_instance = cad2_input.iloc[10].to_numpy()[None, :] #10 is selected randomly

predicted_matrix = model.predict(example_instance) #this is the FCM matrix it has shape (batch_size, concepts, concepts, 1)

### Use the example instance and the weight matrix to perform FCM inference for as many iterations as the model requires
for i in range(1): #change accordigly for more iterations
    example_instance = example_instance + tf.linalg.matmul(example_instance, predicted_matrix[:, :, :, 0])
    example_instance = sigmoid(example_instance, 2) # l = 2
    example_instance = example_instance[:, 0]

prediction = example_instance[0, -2:].numpy() 
prediction_class = np.argmax(prediction)



#### Important info for displaying the matrix
'''
1. The output matrix has shape (batch, num_concepts, num_concepts, 1). slice it to have (concepts, concepts) shape.
2. The matrix diagonal may have small values close to 0, such as 0.01, -0.05 etc. You can use a function such as np.fill_diagonal(predicted_matrix, 0)
    to set these values to 0 for proper graph. 
3. Similarly for the last 2 rows of the matrix. Also these values are desired to be 0 but the learning algorithm provides values close to 0
    You may use:
    predicted_matrix[-2, :] = 0
    predicted_matrix[-1, :] = 0

'''