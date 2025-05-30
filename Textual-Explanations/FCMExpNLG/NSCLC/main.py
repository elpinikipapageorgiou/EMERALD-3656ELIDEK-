#--- IMPORT DEPENDENCIES ------------------------------------------------------+
import numpy as np
import time
from random import uniform
import pandas as pd
from sklearn.model_selection import KFold
import math
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from statistics import mean
from openpyxl import load_workbook
import matplotlib.pyplot as plt
import random


from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx

def wrap_labels(labels, max_length=5):
    wrapped_labels = {}
    for key, label in labels.items():
        words = label.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            if current_length + len(word) + len(current_line) > max_length:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word)
        lines.append(' '.join(current_line))
        wrapped_labels[key] = '\n'.join(lines)
    return wrapped_labels

def sig(x):
    return 1/(1 + np.exp(-x))

def plot_FCM_weight_matrix_graph(column_names, best_position, index,scale_factor=2.5):
    # Determine the output node based on the study

    output_node = 'NSCLC'
    
    best_position = best_position[:-1]
    
    
    
    # if type_data =='multimodal' and study=='CAD':
    #     best_position = best_position[:-1]
    # st.write(type_data, study, best_position, column_names)
    # Create directed graph
    G = nx.DiGraph()
    # st.write(column_names, best_position)
    # Add nodes to the graph
    G.add_nodes_from(column_names + [output_node])

    # Round the best_position weights to two decimals
    best_position = [round(weight, 2) for weight in best_position]

    # Add weighted edges to the graph based on the best_position list
    for i, weight in enumerate(best_position):
        if abs(weight) > 0.01:  # Lower the threshold to include more edges
            G.add_edge(column_names[i], output_node, weight=weight)

    # Remove nodes with no edges
    nodes_to_remove = [node for node in G.nodes if G.degree(node) == 0]
    G.remove_nodes_from(nodes_to_remove)

    # Draw the graph
    fig, ax = plt.subplots(figsize=(12, 12))  # Increase figure size
    pos = nx.circular_layout(G)  # Use circular layout

    # Scale the positions to spread out the nodes
    for key in pos:
        pos[key] = scale_factor * pos[key]

    # Position the output node in the center
    if output_node in pos:
        pos[output_node] = np.array([0, 0])

    # Draw edges with colors, widths, and arrows
    edges = G.edges(data=True)
    weights = [d['weight'] for _, _, d in edges]
    edge_colors = ['red' if w < 0 else 'green' for w in weights]  # Red for negative, Green for positive
    edge_widths = [6 * abs(w) for w in weights]  # Set edge widths based on the absolute value of weights

    nx.draw_networkx_edges(
        G, pos, edgelist=edges, width=edge_widths, edge_color=edge_colors, ax=ax,
        arrows=True, arrowstyle='-|>', arrowsize=70, connectionstyle='arc3,rad=0.2'
    )

    # Draw nodes after edges
    nx.draw_networkx_nodes(G, pos, node_size=7000, node_color='#1F3A63', ax=ax)
    nx.draw_networkx_labels(G, pos, labels=wrap_labels({n: n for n in G.nodes}, max_length=5), font_size=12, font_color='white', font_weight='bold', ax=ax)


    # Add edge labels (weights) closer to the edges with font colors
    edge_labels = {(n1, n2): f"{d['weight']:.2f}" for n1, n2, d in G.edges(data=True)}
    for (n1, n2), label in edge_labels.items():
        font_color = 'red' if G[n1][n2]['weight'] < 0 else 'green'
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels={(n1, n2): label}, font_color=font_color, font_size=12, ax=ax,
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'),
            label_pos=0.7,  # Center the labels
            rotate=False  # Disable rotation to keep labels aligned with the edges
        )

    # Create color legend for weights (green for positive, red for negative)
    if weights:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=-1, vmax=1))  # RdYlGn: green=positive, red=negative
        sm.set_array([])

        # Customize the color bar to match edge colors
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label('Edge Weight', rotation=270, labelpad=20)

    # Save the plot with the unique index
    output_filename = f"fcm_graph_{index}.png"
    fig.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')
    print(f"Saved the plot as: {output_filename}")




def custom_prediction(num_dimensions, best_position, user_inputs, small_value=0.01, weight_threshold=0.1):
    predicted_results = [0] * num_dimensions
    for i in range(num_dimensions):
        sum_temp = sum(best_position[j][i] * user_inputs[j] for j in range(num_dimensions) if i != j)
        
        # Apply sigmoid function to get the predicted result
        predicted_results[i] = sig(sum_temp)
        
        # Check the sign of the element in the last column of best_position for this index i
        last_column_sign = np.sign(best_position[i][-1])  # Get the sign of the last column for the i-th row
        
        # Adjust the predicted result to maintain the same sign as the last column
        if last_column_sign < 0:
            predicted_results[i] = -abs(predicted_results[i])
        else:
            predicted_results[i] = abs(predicted_results[i])

    # The final output prediction is the last element of predicted_results
    output_prediction = predicted_results[-1]

    return output_prediction, predicted_results

def remove_indices_nsclc(user_inputs, user_inputs_dict,best_position ):
        selected_location = f"location {user_inputs_dict['Location'].lower().replace('_', '-')}"
        selected_type = f"type missing" if user_inputs_dict['Type'] == "N/A" else f"type {user_inputs_dict['Type'].lower().replace('_', '-')}"
        selected_margins = f"margins missing" if user_inputs_dict['Margins'] == "N/A" else f"margins {user_inputs_dict['Margins'].lower().replace('_', '-')}"

        
        # st.write(type(best_position))
        column_names = best_position[0, :].tolist()
    
        # Normalize all column names to lowercase for consistency
        column_names_lower = [col.lower().replace('_', '-') for col in column_names]
        if 'BMI' in user_inputs_dict and user_inputs_dict['BMI'] is None:
    
            # Find the index of the 'BMI' column in the column_names list
            bmi_column_index = column_names.index("BMI")
            # print(f"Index of BMI column: {bmi_column_index}")
            
            # Delete the 'BMI' column and the row with the same index from best_position
            best_position = np.delete(best_position, bmi_column_index, axis=1)  # Drop the column
            bmi_column_index = bmi_column_index +1
            best_position = np.delete(best_position, bmi_column_index, axis=0)  # Drop the row with the same index
            # print("Updated best_position after removing BMI column and row:")
            

        column_names = best_position[0, :].tolist()
        
        # Normalize all column names to lowercase for consistency
        column_names_lower = [col.lower().replace('_', '-') for col in column_names]
        
        # Initialize the list for deleted indices
        deleted_indices = []


        # Get the indices of all location-related columns that do not match the selected location
        for i, col in enumerate(column_names_lower):
            if col.startswith("location") and col != selected_location:
                deleted_indices.append(i)

        # If "type missing" is selected, remove all type-related columns
        if user_inputs_dict['Type'] == "N/A":
            for i, col in enumerate(column_names_lower):
                if col.startswith("type"):
                    deleted_indices.append(i)
        else:
            # If a specific type is selected, only remove other type-related columns
            for i, col in enumerate(column_names_lower):
                if col.startswith("type") and col != selected_type:
                    deleted_indices.append(i)

        # If "margins missing" is selected, remove all margin-related columns
        if user_inputs_dict['Margins'] == "N/A":
            for i, col in enumerate(column_names_lower):
                if col.startswith("margins"):
                    deleted_indices.append(i)
        else:
            # If a specific margin is selected, only remove other margin-related columns
            for i, col in enumerate(column_names_lower):
                if col.startswith("margins") and col != selected_margins:
                    deleted_indices.append(i)


        
        best_position_filtered = np.delete(best_position, deleted_indices, axis=1)  # Delete columns

        deleted_indices_rows = [index + 1 for index in deleted_indices]
        best_position = np.delete(best_position_filtered, deleted_indices_rows, axis=0)  # Delete rows
        


        if not np.all(best_position[-1] == 0):
            # If not, add a row of zeros
            zero_row = np.zeros(best_position.shape[1], dtype=int)  # Create a row of zeros with the same number of columns
            best_position = np.vstack([best_position, zero_row])
        # st.write(user_inputs_dict)
        if 'Gender' in user_inputs_dict:
            best_position[1, -1] =  random.uniform(0.1, 0.25)
        
        if 'Age' in user_inputs_dict:
            best_position[2, -1] =  random.uniform(0.1, 0.25)

        if 'CNN' in user_inputs_dict:
            cnn_index = list(user_inputs_dict.keys()).index('CNN')
            cnn_index +=1
        
            if user_inputs_dict.get('CNN', None) == 0:
                min_val = -0.05
                max_val = -0.01
                # If it is, replace the last column of the 5th row (index 4)
                random_number = random.uniform(min_val, max_val)
                best_position[-2, -1] = random_number
            else:
                best_position[-2, -1] = best_position[-2, -1]

        for index in sorted(deleted_indices, reverse=True):
            user_inputs.pop(index)
        
        
        # st.write(user_inputs)
        num_dimensions = len(user_inputs) + 1 
        if user_inputs_dict['BMI']==None:
            num_dimensions = len(user_inputs) 
            user_inputs = [element for element in user_inputs if element is not None]
            
        column_names = best_position[0, :].tolist()
        # st.write(best_position)
        return user_inputs, num_dimensions, column_names, best_position

def generate_factual_explanation(input_data, interconnection_strength_text, classification):
    explanation = []

    # Static text for both "malignant" and "normal" classification
    explanation.append("Based on the input values provided for the patient case, the Fuzzy Cognitive Map (FCM) has determined the classification ")

    # Add the introductory statement based on classification
    if classification == "malignant":
        explanation.append("as malignant, primarily due to key clinical risk factors indicating a higher likelihood of cancer or disease progression.")
    else:
        explanation.append("as normal, suggesting a lower likelihood of disease progression, though some clinical risk factors still need monitoring.")

    # Identify factors with non-zero values and strong or very strong impact
    high_impact_factors = []
    medium_impact_factors = []
    # print(input_data, interconnection_strength_text)
    if classification =='malignant':
        for feature, strength in interconnection_strength_text.items():
            
            if strength in ["Strong", "Very Strong", "Medium"] and input_data.get(feature) != 0:
                # print(feature,strength,  input_data.get(feature))
                # Group features for malignant classification
                if feature == "SUV" and input_data.get(feature) != 0:
                    high_impact_factors.append("• High SUV: A higher Standardized Uptake Value (SUV) in PET scans may indicate increased metabolic activity, often associated with malignancy.")
                elif feature == "Type Ground-Class" and input_data.get(feature) == 1:
                    medium_impact_factors.append("• Ground class type: Ground glass opacity may indicate a higher likelihood of malignancy in some cases.")
                elif feature == "Type Semi-Solid" and input_data.get(feature) == 1:
                    medium_impact_factors.append("• Semi-solid type: Semi-solid nodules can be concerning as they may be indicative of malignancy.")
                elif feature == "Type cavitary" and input_data.get(feature) == 1:
                    medium_impact_factors.append("• Cavitary type: Cavitary lesions can be indicative of malignancy, especially when associated with certain symptoms.")
                elif feature == "Type Consolidated" and input_data.get(feature) == 1:
                    medium_impact_factors.append(
                        "• Consolidated type: Consolidation in imaging may suggest a more advanced or aggressive lesion, often associated with malignancy in NSCLC cases."
                    )
                elif feature == "Diameter" and input_data.get(feature) >0.01:
                    high_impact_factors.append("• Large diameter: Larger tumor size often correlates with a higher risk of malignancy and more aggressive disease.")
                elif feature == "Margins Spiculated" and input_data.get(feature) == 1:
                    high_impact_factors.append("• Spiculated margins: Spiculated margins are a common characteristic of malignant tumors and suggest a higher likelihood of cancer.")
                elif feature == "Margins Ill-Defined" and input_data.get(feature) == 1:
                    medium_impact_factors.append("• Ill-defined margins: Ill-defined margins are more common in malignant tumors and may indicate aggressive growth.")
                elif feature == "Margins Lobulated" and input_data.get(feature) == 1:
                    medium_impact_factors.append("• Lobulated margins: Lobulated margins can suggest the presence of malignancy, especially when combined with other features.")
                elif feature == "Location Left Lower Lobe" and input_data.get(feature) == 1:
                    high_impact_factors.append("• Location in the left lower lobe: The location of a tumor in the left lower lobe is typically associated with certain cancer types, which may influence prognosis.")
                elif feature == "Location Right Upper Lobe" and input_data.get(feature) == 1:
                    medium_impact_factors.append("• Location in the right upper lobe: While generally less concerning than other locations, tumors in the right upper lobe still warrant careful evaluation.")
                elif feature == "Location Right lower lobe" and input_data.get(feature) == 1:
                    medium_impact_factors.append("• Location in the right lower lobe: Tumors in the right lower lobe can be concerning depending on their size and characteristics.")
                elif feature == "Location Left upper lobe" and input_data.get(feature) == 1:
                    high_impact_factors.append("• Location in the left upper lobe: Tumors in the left upper lobe are more commonly associated with malignancy and require urgent investigation.")
                elif feature == "Location Middle" and input_data.get(feature) == 1:
                    medium_impact_factors.append("• Location in the middle: Middle lobe tumors, although less common, can still present significant risks depending on their size and features.")
                elif feature == "Location Lingula" and input_data.get(feature) == 1:
                    medium_impact_factors.append("• Location in the lingula: Tumors in the lingula require careful assessment as they can have diverse clinical presentations.")
                elif feature == "CNN" and input_data.get(feature) == 1:
                    high_impact_factors.append(
                        "• CNN Prediction: The convolutional neural network (CNN) has identified this instance as malignant based solely on imaging data. This AI-driven assessment provides a critical factor in NSCLC diagnosis, offering an objective analysis that complements clinical evaluation."
                    )
        # Add the high-impact and medium-impact factors as bullet points
        # print(high_impact_factors)
        if high_impact_factors:
            explanation.append("The most significant contributing factors for malignancy include:")
            for factor in high_impact_factors:
                explanation.append(factor)

        if medium_impact_factors:
            explanation.append("Other factors to consider include:")
            for factor in medium_impact_factors:
                explanation.append(factor)

    if classification =='benign':
        for feature, strength in interconnection_strength_text.items():
            
            if strength in ["negative Strong", "Strong","negative Very Strong", "negative Medium", "Very Strong", "Medium"] and input_data.get(feature) != 0:
                if feature == "Margins Well Defined" and input_data.get(feature) != 0:
                    high_impact_factors.append("• Well-defined margins: Tumors with well-defined margins are typically associated with benign growth, suggesting a more localized and less aggressive nature.")
                elif feature == "Type Speckled" and input_data.get(feature) == 1:
                    high_impact_factors.append("• Speckled type: This type of tumor is often observed in benign cases, indicating a more stable and non-invasive nature.")
                elif feature == "Type Calcified" and input_data.get(feature) == 1:
                    medium_impact_factors.append("• Calcified type: Calcification in tumors is commonly seen in benign growths, often indicating older, non-aggressive lesions.")
                elif feature == "Type Solid" and input_data.get(feature) == 1:
                    medium_impact_factors.append("• Solid type: Solid tumors, while warranting careful monitoring, can sometimes be benign, especially when associated with stable characteristics.")

                if high_impact_factors:
                    explanation.append("The most significant contributing factors for the benign tumor include:")
                    for factor in high_impact_factors:
                        explanation.append(factor)


        if medium_impact_factors:
            explanation.append("Other factors to consider include:")
            for factor in medium_impact_factors:
                explanation.append(factor)
        
    
    return "\n".join(explanation)

def convert_to_strength(value):
    # Handle positive values first
    if value >= 0:
        if 0 <= value <= 0.25:
            return "Very Weak"
        if 0.1 <= value <= 0.4:
            return "Weak"
        if 0.55 <= value <= 0.85:
            return "Strong"
        if 0.35 <= value <= 0.65:
            return "Medium"
        
        if 0.75 <= value <= 1:
            return "Very Strong"
    else:
        # Handle negative values by inverting the thresholds
        if -0.25 <= value < 0:
            return "negative Very Weak"
        if -0.4 <= value < -0.1:
            return "negative Weak"
        if -0.65 <= value < -0.35:
            return "negative Medium"
        if -0.85 <= value < -0.55:
            return "negative Strong"
        if value < -0.75:
            return "negative Very Strong"



#--- MAIN ---------------------------------------------------------------------

#read_clinical_data
dataset=pd.read_excel("dataset.xlsx", engine='openpyxl')
initial_dataset = dataset.copy()

column_names = dataset.columns[:-1]
dataset['DIAMETER'] = dataset['DIAMETER'] / 3
dataset['BMI'] = dataset['BMI'] / 70
dataset['SUV'] = dataset['SUV'] / 30
dataset['GLU'] = dataset['GLU'] / 192

column = 'AGE'
dataset[column] = (dataset[column]-dataset[column].min())/(dataset[column].max()-dataset[column].min())

dataset["Location Left upper lobe"] = np.where(dataset["Location"] =='Left upper lobe', 1, 0)
dataset["Location Right upper lobe"] = np.where(dataset["Location"] =='Right upper lobe', 1, 0)
dataset["Location Left lower lobe"] = np.where(dataset["Location"] =='Left lower lobe', 1, 0)
dataset["Location Right lower lobe"] = np.where(dataset["Location"] =='Right lower lobe', 1, 0)
dataset["Location MIDDLE"] = np.where(dataset["Location"] =='MIDDLE', 1, 0)
dataset["Location LINGULA"] = np.where(dataset["Location"] =='LINGULA', 1, 0)
dataset= dataset.drop(['Location'], axis=1)

dataset["Type MISSING"] =  np.where(dataset["Type"].isna(), 1, 0)
dataset["Type SOLID"] = np.where(dataset["Type"] =='SOLID', 1, 0)
dataset["Type GROUND_CLASS"] = np.where(dataset["Type"] =='GROUND-CLASS', 1, 0)
dataset["Type CONSOLIDATED"] = np.where(dataset["Type"] =='CONSOLIDATED', 1, 0)
dataset["Type SPECKLED"] = np.where(dataset["Type"] =='SPECKLED', 1, 0)
dataset["Type SEMI_SOLID"] = np.where(dataset["Type"] =='SEMI-SOLID', 1, 0)
dataset["Type calcified"] = np.where(dataset["Type"] =='calcified', 1, 0)
dataset["Type cavitary"] = np.where(dataset["Type"] =='cavitary', 1, 0)
dataset= dataset.drop(['Type'], axis=1)

dataset["Margins missing"] = np.where(dataset["Margins"].isna(), 1, 0)
dataset["Margins spiculated"] = np.where(dataset["Margins"] =='spiculated', 1, 0)
dataset["Margins lobulated"] = np.where(dataset["Margins"] =='lobulated', 1, 0)
dataset["Margins well_defined"] = np.where(dataset["Margins"] =='well defined', 1, 0)
dataset["Margins ill-defined"] = np.where(dataset["Margins"] =='ill-defined', 1, 0)
dataset= dataset.drop(['Margins'], axis=1)



dataset= dataset.drop(['id'], axis=1)

column_name = 'Output'
# Extract the column
selected_column = dataset[column_name]
# Drop the column from its current position
dataset = dataset.drop(column_name, axis=1)
# Re-insert the column at the end
dataset[column_name] = selected_column
dataset.fillna(method="bfill", inplace=True)


num_dimensions = len(dataset.columns) 

# Normalize only 'AGE' and 'BMI' columns, applying to numeric values
columns_to_normalize = ['AGE', 'BMI']
for column in columns_to_normalize:
    if column in dataset.columns:  # Check if the column exists in the dataset
        # Ensure that the column contains numeric values
        if pd.api.types.is_numeric_dtype(dataset[column]):
            unique_vals = dataset[column].nunique()  # Get the number of unique values
            if unique_vals > 1:  # If there is more than one unique value, normalize
                min_val = dataset[column].min()
                max_val = dataset[column].max()
                # Check to prevent division by zero
                if max_val != min_val:
                    dataset[column] = (dataset[column] - min_val) / (max_val - min_val)
                else:
                    dataset[column] = 0.5  # Assign middle value if there's only one unique value
            else:
                dataset[column] = 0.5  # For single unique value, set it to the middle point (0.5)
                # print(f"Column {column} has only one unique value, normalization skipped.")
        # else:
        #     print(f"Column {column} is not numeric, skipping normalization.")



df = pd.read_excel("mean_values.xlsx", header=None)
best_position = df.to_numpy()
user_inputs_dict = {}


# If multiple rows, process each row
for index, row in dataset.iterrows():
    
    df = pd.read_excel("mean_values.xlsx", header=None)
    best_position = df.to_numpy()
    # Prepare the input for each row (second_row)
    user_inputs = row.tolist()  # Convert row to a list


    # print(dataset)
    user_inputs_dict['Gender'] = initial_dataset['Gender'].iloc[index]
    user_inputs_dict['AGE'] = initial_dataset['AGE'].iloc[index]
    user_inputs_dict['BMI'] = initial_dataset['BMI'].iloc[index]
    user_inputs_dict['GLU'] = initial_dataset['GLU'].iloc[index]
    user_inputs_dict['SUV'] = initial_dataset['SUV'].iloc[index]
    user_inputs_dict['DIAMETER'] = initial_dataset['DIAMETER'].iloc[index]
    user_inputs_dict['Location'] = initial_dataset['Location'].iloc[index]
    user_inputs_dict['Type'] = initial_dataset['Type'].iloc[index] if pd.notna(initial_dataset['Type'].iloc[index]) else 'N/A'
    user_inputs_dict['Margins'] = initial_dataset['Margins'].iloc[index] if pd.notna(initial_dataset['Margins'].iloc[index]) else 'N/A'
    

    # print(user_inputs_dict)
    # exit()
    
    # print(len(user_inputs), user_inputs_dict, best_position.shape, len(column_names), (num_dimensions))
    user_inputs, num_dimensions, column_names, best_position = remove_indices_nsclc(user_inputs, user_inputs_dict,best_position )

    best_position = pd.DataFrame(best_position)
    best_position.iloc[0, -1] = 'NSCLC'
    if (best_position.iloc[-1] == 0).all() and (best_position.iloc[-2] == 0).all():
        # Drop the last row
        best_position = best_position.drop(best_position.index[-1])

    column_names[-1] = 'NSCLC'
    best_position = best_position.to_numpy()
    best_position = best_position[1:]
    best_position = np.array(best_position)
    flattened_list = []
    user_inputs_list_of_lists=user_inputs

    for item in user_inputs_list_of_lists:
        if isinstance(item, list):
            flattened_list.extend(item)
        else:
            flattened_list.append(item)
    user_inputs = flattened_list
    # Read the best position matrix
    # user_inputs.append(0.5)
    # print(best_position)
    num_dimensions = num_dimensions -1
    # print(num_dimensions, len(user_inputs), len(column_names), len(best_position))
    
    # Apply the custom_prediction function
    output_prediction, predicted_results = custom_prediction(num_dimensions=num_dimensions, 
                                                                best_position=best_position, 
                                                                user_inputs=user_inputs)
    
    # Print the result for each instance (each row in the dataset)
    print(f"\n***************\nPatient {index + 1}:")

    if output_prediction > 0.74:
        classification = "malignant"
        print("The patient is likely to have NSCLC")
    else:
        classification = "benign"
        print("The patient is not likely to have NSCLC")
    # print(column_names, predicted_results)
    # Plot FCM Weight Matrix for the current row
    plot_FCM_weight_matrix_graph(column_names=column_names, index=index,
                                    best_position=predicted_results)
    column_names.pop()
    predicted_results.pop()
    dict_results = dict(zip(column_names, predicted_results))
    
    # Convert the numeric interconnections to textual descriptions
    interconnection_strength_text = {key: convert_to_strength(value) for key, value in dict_results.items()}
    input_data = dict(zip(column_names, user_inputs))

    input_data['Gender'] = user_inputs_dict['Gender']
    input_data['AGE'] = user_inputs_dict['AGE']
    input_data['BMI'] = user_inputs_dict['BMI']
    input_data['GLU'] = user_inputs_dict['GLU']
    input_data['SUV'] = user_inputs_dict['SUV']
    input_data['Diameter'] = user_inputs_dict['DIAMETER']
    

    factual_explanation = generate_factual_explanation(input_data, interconnection_strength_text, classification)


    # Output the explanations
    print("\nFactual Explanation:")
    print(factual_explanation)
