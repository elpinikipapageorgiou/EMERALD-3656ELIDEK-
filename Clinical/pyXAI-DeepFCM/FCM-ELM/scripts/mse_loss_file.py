import numpy as np


def mse_loss(training_real_output, current_fcm_output):
    sum_squared_errors = np.sum((training_real_output - current_fcm_output) ** 2)
    mse = sum_squared_errors / len(training_real_output)
    return mse

def mse_loss_with_regularization(training_real_output, current_fcm_output, weights, initial_weights, lambda_reg):
    mse = mse_loss(training_real_output, current_fcm_output)
    regularization_term = lambda_reg * np.sum((weights - initial_weights) ** 2)
    return mse + regularization_term
