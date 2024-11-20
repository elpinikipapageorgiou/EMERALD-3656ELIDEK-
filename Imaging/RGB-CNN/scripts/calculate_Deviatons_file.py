# Import important libraries
import math


def calculate_deviation(values):
    # Step 1
    mean = sum(values) / len(values)
    
    # Step 2
    differences = [value - mean for value in values]
    
    # Step 3
    squared_differences = [diff ** 2 for diff in differences]
    
    # Step 4
    mean_squared_differences = sum(squared_differences) / len(squared_differences)
    
    # Step 5
    standard_deviation = math.sqrt(mean_squared_differences)
    
    return standard_deviation

def calculate_npv(confusion_matrix):
    true_negatives = confusion_matrix[0][0]
    false_negatives = confusion_matrix[1][0]
    npv = true_negatives / (true_negatives + false_negatives)
    return npv

def calculate_tpr(confusion_matrix):
    # Extract values from confusion matrix
    tp = confusion_matrix[1][1]  # True Positives
    fn = confusion_matrix[1][0]  # False Negatives
    
    # Calculate TPR
    tpr = tp / (tp + fn)
    return tpr