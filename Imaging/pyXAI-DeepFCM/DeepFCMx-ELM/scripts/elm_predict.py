import numpy as np

def elm_predict_function(input_data, input_weights, output_weights):
    hidden_layer_output = np.dot(input_data, input_weights.T)
    output_weights = output_weights[:-1]
    predicted_output = np.dot(hidden_layer_output, output_weights)
    return predicted_output, hidden_layer_output