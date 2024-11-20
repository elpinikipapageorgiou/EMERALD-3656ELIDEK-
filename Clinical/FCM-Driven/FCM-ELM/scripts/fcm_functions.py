import numpy as np
from elm_predict_file import elm_predict
from mse_loss_file import mse_loss_with_regularization

# Define your sigmoid function
def sig(x):
    return 1 / (1 + np.exp(-x))

def update_fcm_weights(weights, input_features):
    num_dimensions = weights.shape[0]
    updated_weights = np.zeros_like(weights)
    
    for row in input_features:
        for i in range(num_dimensions):
            sum_temp = 0
            for j in range(num_dimensions):
                if i != j:
                    sum_temp += weights[j][i] * row[j]
            updated_weights[i] = sig(row[i] + sum_temp)
    
    return updated_weights

def constrain_weights(weights, initial_weights, lower_bounds, upper_bounds):
    constrained_weights = np.clip(weights, lower_bounds, upper_bounds)
    return constrained_weights

def FCM_ELM(initial_weights, num_iterations, input_features, num_hidden_units=30,learning_rate=0.1, lambda_reg=0.01):
    best_weights = initial_weights.copy()
    best_mse = float('inf')
    num_dimensions = initial_weights.shape[0]
    np.fill_diagonal(initial_weights, 0)

    training_real_output = input_features[:, -1]  # Assuming the last column is the output

    lower_bounds = initial_weights - 0.4
    upper_bounds = initial_weights + 0.4

    for iteration in range(num_iterations):
        fcm_weights = update_fcm_weights(best_weights, input_features)
        fcm_weights = constrain_weights(fcm_weights, initial_weights, lower_bounds, upper_bounds)
        input_weights = np.random.uniform(size=(num_hidden_units, num_dimensions - 1), low=-1, high=1)
        hidden_layer_output = np.dot(input_features[:, :-1], input_weights.T)  # Exclude the last column for input features
        
        # Add bias term to hidden layer output
        hidden_layer_output_with_bias = np.hstack((hidden_layer_output, np.ones((hidden_layer_output.shape[0], 1))))
        
        # Compute output weights using Moore-Penrose pseudoinverse
        output_weights = np.linalg.pinv(hidden_layer_output_with_bias).dot(training_real_output)
        
        elm_output, _ = elm_predict(input_features[:, :-1], input_weights, output_weights)  # Exclude the last column for input features
        current_mse = mse_loss_with_regularization(training_real_output, elm_output, fcm_weights, initial_weights, lambda_reg)

        if current_mse < best_mse:
            best_mse = current_mse
            best_weights = fcm_weights
        
        if current_mse < 0.8:
            break
        
        best_weights = fcm_weights + learning_rate * ((training_real_output - elm_output) * elm_output * (1 - elm_output)).T @ input_features[:, :-1]


    np.fill_diagonal(best_weights, 0)

    return best_weights

