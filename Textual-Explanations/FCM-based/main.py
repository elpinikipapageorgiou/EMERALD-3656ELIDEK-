#--- IMPORT DEPENDENCIES ------------------------------------------------------+
import numpy as np
import pandas as pd
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
    output_node = 'output'
    
    best_position = best_position[:-1]

    # Create directed graph
    G = nx.DiGraph()

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

    # Calculate individual concept contributions and update predicted_results
    for i in range(num_dimensions):
        sum_temp = sum(best_position[j][i] * user_inputs[j] for j in range(num_dimensions))
        sig_output = sig(sum_temp)
        predicted_results[i] = sig_output if best_position[i][-1] >= 0 else -sig_output

    # Adjust final output prediction based on significant contributions
    significant_contribution = False
    contribution_sum = 0

    for i in range(num_dimensions):
        if user_inputs[i] == 1 and abs(best_position[i][-1]) > weight_threshold:
            significant_contribution = True
            contribution_sum += best_position[i][-1] * user_inputs[i]

    if significant_contribution and contribution_sum > 0:
        output_prediction = min(1, sig(contribution_sum))
    else:
        output_prediction = max(0, sig(contribution_sum))

    predicted_results[-1] = output_prediction

    return output_prediction, predicted_results



def remove_indices(user_inputs_dict, best_position, user_inputs):
    
    # Identify indices with None values in user_inputs_dict
    none_value_indices = [i for i, (key, value) in enumerate(user_inputs_dict.items()) if value is None]

    # Remove rows and columns based on none_value_indices, skipping the first row
    # Remove rows (excluding the first row containing column names)
    best_position = np.delete(best_position, [i + 1 for i in none_value_indices], axis=0)

    # Remove columns based on none_value_indices (including from the first row)
    best_position = np.delete(best_position, none_value_indices, axis=1)

    # st.write(user_inputs_dict, best_position)
    # Update user_inputs_dict and user_inputs to exclude None values
    user_inputs_dict = {key: value for key, value in user_inputs_dict.items() if value is not None}
    user_inputs = [value for i, value in enumerate(user_inputs) if i not in none_value_indices]

    num_dimensions = len(user_inputs) + 1
    # st.write(user_inputs_dict)
    if 'CNN' in user_inputs_dict:
        cnn_index = list(user_inputs_dict.keys()).index('CNN')
        cnn_index +=1
    
        if user_inputs_dict.get('CNN', None) == 0:
            min_val = -0.05
            max_val = -0.01
            # If it is, replace the last column of the 5th row (index 4)
            random_number = random.uniform(min_val, max_val)
            best_position[cnn_index, -1] = random_number
        else:
            best_position[cnn_index, -1] = best_position[cnn_index, -1]
       
    if 'known CAD' in user_inputs_dict:
        known_cad_index = list(user_inputs_dict.keys()).index('known CAD')
        known_cad_index +=1
    
        if user_inputs_dict.get('known CAD', None) == 0:
            
            min_val = -0.05
            max_val = -0.01
            # If it is, replace the last column of the 5th row (index 4)
            random_number = random.uniform(min_val, max_val)
            best_position[known_cad_index, -1] = random_number
        else:
            best_position[known_cad_index, -1] = best_position[known_cad_index, -1]

    
    if 'previous AMI' in user_inputs_dict:
        previous_ami_index = list(user_inputs_dict.keys()).index('previous AMI')
        previous_ami_index +=1

        if user_inputs_dict.get('previous AMI', None) == 0:
            min_val = -0.05
            max_val = -0.01
            # If it is, replace the last column of the 5th row (index 4)
            random_number = random.uniform(min_val, max_val)
            best_position[previous_ami_index, -1] = random_number
        else:
            best_position[previous_ami_index, -1] = best_position[previous_ami_index, -1]
    
    if 'previous PCI' in user_inputs_dict:
        previous_pci_index = list(user_inputs_dict.keys()).index('previous PCI')
        previous_pci_index +=1

        if user_inputs_dict.get('previous PCI', None) == 0:
            min_val = -0.05
            max_val = -0.01
            # If it is, replace the last column of the 5th row (index 4)
            random_number = random.uniform(min_val, max_val)
            best_position[previous_pci_index, -1] = random_number
        else:
            best_position[previous_pci_index, -1] = best_position[previous_pci_index, -1]

    if 'previous CABG' in user_inputs_dict:
        previous_cabg_index = list(user_inputs_dict.keys()).index('previous CABG')
        previous_cabg_index +=1
        if user_inputs_dict.get('previous CABG', None) == 0:
            min_val = -0.05
            max_val = -0.01
            # If it is, replace the last column of the 5th row (index 4)
            random_number = random.uniform(min_val, max_val)
            best_position[previous_cabg_index, -1] = random_number
        else:
            best_position[previous_cabg_index, -1] = best_position[previous_cabg_index, -1]

    if 'previous Stroke' in user_inputs_dict:
        previous_stroke_index = list(user_inputs_dict.keys()).index('previous Stroke')
        previous_stroke_index +=1
        if user_inputs_dict.get('previous Stroke', None) == 0:
            min_val = -0.05
            max_val = -0.01
            # If it is, replace the last column of the 5th row (index 4)
            random_number = random.uniform(min_val, max_val)
            best_position[previous_stroke_index, -1] = random_number
        else:
            best_position[previous_stroke_index, -1] = best_position[previous_stroke_index, -1]
    
    if 'Diabetes' in user_inputs_dict:
        diabetes_index = list(user_inputs_dict.keys()).index('Diabetes')
        diabetes_index +=1
        if user_inputs_dict.get('Diabetes', None) == 0:
            min_val = -0.2
            max_val = -0.1
            # If it is, replace the last column of the 5th row (index 4)
            random_number = random.uniform(min_val, max_val)
            best_position[diabetes_index, -1] = random_number
        else:
            best_position[diabetes_index, -1] = best_position[diabetes_index, -1]

    if 'Smoking' in user_inputs_dict:
        smoking_index = list(user_inputs_dict.keys()).index('Smoking')
        smoking_index +=1
        if user_inputs_dict.get('Smoking', None) == 0:
            min_val = -0.2
            max_val = -0.1
            # If it is, replace the last column of the 5th row (index 4)
            random_number = random.uniform(min_val, max_val)
            best_position[smoking_index, -1] = random_number
        else:
            best_position[smoking_index, -1] = best_position[smoking_index, -1]
    
    if 'Arterial Hypertension' in user_inputs_dict:
        hypertension_index = list(user_inputs_dict.keys()).index('Arterial Hypertension')
        hypertension_index +=1
        if user_inputs_dict.get('Arterial Hypertension', None) == 0:
            min_val = -0.15
            max_val = -0.1
            # If it is, replace the last column of the 5th row (index 4)
            random_number = random.uniform(min_val, max_val)
            best_position[hypertension_index, -1] = random_number
        else:
            best_position[hypertension_index, -1] = best_position[hypertension_index, -1]

    if 'Dyslipidemia' in user_inputs_dict:
        dislipidemia_index = list(user_inputs_dict.keys()).index('Dyslipidemia')
        dislipidemia_index +=1
        if user_inputs_dict.get('Dyslipidemia', None) == 0:
            min_val = -0.05
            max_val = -0.01
            # If it is, replace the last column of the 5th row (index 4)
            random_number = random.uniform(min_val, max_val)
            best_position[dislipidemia_index, -1] = random_number
        else:
            best_position[dislipidemia_index, -1] = best_position[dislipidemia_index, -1] 

    if 'Angiopathy' in user_inputs_dict:
        angiopathy_index = list(user_inputs_dict.keys()).index('Angiopathy')
        angiopathy_index +=1
        if user_inputs_dict.get('Angiopathy', None) == 0:
            min_val = -0.05
            max_val = -0.01
            # If it is, replace the last column of the 5th row (index 4)
            random_number = random.uniform(min_val, max_val)
            best_position[angiopathy_index, -1] = random_number
        else:
            best_position[angiopathy_index, -1] = best_position[angiopathy_index, -1]

    if 'Chronic Kidney Disease' in user_inputs_dict:
        chronic_kidney_index = list(user_inputs_dict.keys()).index('Chronic Kidney Disease')
        chronic_kidney_index +=1
        if user_inputs_dict.get('Chronic Kidney Disease', None) == 0:
            min_val = -0.05
            max_val = -0.01
            # If it is, replace the last column of the 5th row (index 4)
            random_number = random.uniform(min_val, max_val)
            best_position[chronic_kidney_index, -1] = random_number
        else:
            best_position[chronic_kidney_index, -1]= best_position[chronic_kidney_index, -1]

    if 'Family History of CAD' in user_inputs_dict:
        family_history_of_cad_index = list(user_inputs_dict.keys()).index('Family History of CAD')
        family_history_of_cad_index +=1
        if user_inputs_dict.get('Family History of CAD', None) == 0:
            min_val = -0.05
            max_val = -0.01
            # If it is, replace the last column of the 5th row (index 4)
            random_number = random.uniform(min_val, max_val)
            best_position[family_history_of_cad_index, -1] = random_number
        else:
            best_position[family_history_of_cad_index, -1] = best_position[family_history_of_cad_index, -1]
    
    if 'ECG' in user_inputs_dict:
        ecg_index = list(user_inputs_dict.keys()).index('ECG')
        ecg_index +=1
        if user_inputs_dict.get('ECG', None) == 0:
            min_val = -0.3
            max_val = -0.2
            # If it is, replace the last column of the 5th row (index 4)
            random_number = random.uniform(min_val, max_val)
            best_position[ecg_index, -1] = random_number
        else:
            best_position[ecg_index, -1] = best_position[ecg_index, -1]
        
    if not np.all(best_position[-1] == 0):
        # If not, add a row of zeros
        zero_row = np.zeros(best_position.shape[1], dtype=int)  # Create a row of zeros with the same number of columns
        best_position = np.vstack([best_position, zero_row])

    return num_dimensions, user_inputs_dict, best_position, user_inputs

# Factual explanation generation function
def generate_factual_explanation(input_data, interconnection_strength_text, classification):
    explanation = []

    # Static text for both "malignant" and "normal" classification
    explanation.append("Based on the input values provided for the patient case, the Fuzzy Cognitive Map (FCM) has determined the classification ")

    # Add the introductory statement based on classification
    if classification == "pathological":
        explanation.append("as pathological, primarily due to key clinical risk factors indicating a higher likelihood of CAD.")
    else:
        explanation.append("as normal, suggesting a lower likelihood of CAD, though some clinical risk factors still need monitoring.")

    # Identify factors with non-zero values and strong or very strong impact
    high_impact_factors = []
    medium_impact_factors = []
    if classification == "pathological":
        for feature, strength in interconnection_strength_text.items():
            if strength in ["Strong", "Very Strong", "Medium"] and input_data.get(feature) != 0:
                # Add feature-based conditions with corresponding explanations
                if feature == "known CAD" and input_data.get(feature) == 1:
                    high_impact_factors.append("• a history of known CAD: Having a history of coronary artery disease significantly increases the risk of further heart issues, suggesting ongoing or past cardiovascular conditions.")
                elif feature == "previous AMI" and input_data.get(feature) == 1:
                    high_impact_factors.append("• a history of previous AMI (Acute Myocardial Infarction): A prior heart attack increases the likelihood of future heart-related issues and worsens the risk of CAD.")
                elif feature == "Smoking" and input_data.get(feature) == 1:
                    high_impact_factors.append("• smoking: Smoking accelerates atherosclerosis, narrowing the arteries and increasing the risk of coronary artery disease.")
                elif feature == "Angiopathy" and input_data.get(feature) == 1:
                    high_impact_factors.append("• the presence of angiopathy: Angiopathy, often seen in conditions like diabetes and hypertension, can damage the blood vessels, contributing to CAD development.")
                elif feature == "Diabetes" and input_data.get(feature) == 1:
                    high_impact_factors.append("• the presence of diabetes: Diabetes is a strong risk factor for CAD, as it can lead to plaque buildup in the arteries and worsen vascular health.")
                elif feature == "Hypertension" and input_data.get(feature) == 1:
                    high_impact_factors.append("• the presence of hypertension: High blood pressure puts additional strain on the heart and arteries, significantly contributing to the progression of CAD.")
                elif feature == "Family History of CAD" and input_data.get(feature) == 1:
                    high_impact_factors.append("• a family history of CAD: A genetic predisposition to CAD increases the likelihood of developing heart-related conditions due to inherited risk factors.")
                elif feature == "previous PCI" and input_data.get(feature) == 1:
                    high_impact_factors.append("• a history of previous PCI (Percutaneous Coronary Intervention): Having undergone PCI suggests previous artery blockages and increases the risk of recurring heart conditions.")
                elif feature == "previous CABG" and input_data.get(feature) == 1:
                    high_impact_factors.append("• a history of previous CABG (Coronary Artery Bypass Grafting): Previous CABG indicates severe coronary artery disease, which increases the risk of future cardiovascular events.")

                # Add explanations for Medium and Very Strong for Normal classification
                elif feature == "Dyslipidemia" and input_data.get(feature) == 1:
                    medium_impact_factors.append("• dyslipidemia: While dyslipidemia is a risk factor for CAD, managing cholesterol levels can help reduce risk.")
                elif feature == "Chronic Kidney Disease" and input_data.get(feature) == 1:
                    medium_impact_factors.append("• chronic kidney disease: Kidney disease often coexists with CAD, and managing both conditions together improves cardiovascular health.")
                elif feature == "Atypical Symptoms" and input_data.get(feature) == 1:
                    medium_impact_factors.append("• atypical symptoms: The presence of atypical symptoms can indicate that the CAD is not presenting in its usual form.")
                elif feature == "Angina Like" and input_data.get(feature) == 1:
                    medium_impact_factors.append("• angina-like symptoms: Angina-like symptoms are often associated with CAD, though not all cases present with typical chest pain.")
                elif feature == "Dyspnoea on Exertion" and input_data.get(feature) == 1:
                    medium_impact_factors.append("• dyspnoea on exertion: Shortness of breath on exertion can be an early sign of heart disease or other cardiovascular issues.")
    
        # Add the high-impact factors as bullet points
        if high_impact_factors:
            explanation.append("The most significant contributing factors for malignancy include:")
            for factor in high_impact_factors:
                explanation.append(factor)

        if medium_impact_factors:
            explanation.append("Other factors to consider include:")
            for factor in medium_impact_factors:
                explanation.append(factor)

        return "\n".join(explanation)

    # If classification is normal, suggest mitigating or less impactful factors
      # Add mitigating factors for normal classification (low risk)
    elif classification == "normal":
        explanation.append("\nFor a normal classification, the following factors are contributing to a lower likelihood of CAD:")
        if input_data.get("Smoking") == 0:
            explanation.append("• smoking: The absence of smoking reduces the risk of heart disease.")
        if input_data.get("Hypertension") == 0:
            explanation.append("• hypertension: Effective management of hypertension lowers cardiovascular risk.")
        if input_data.get("Family History of CAD") == 0:
            explanation.append("• family history of CAD: Lack of a genetic predisposition to CAD further reduces the likelihood.")
        if input_data.get("Diabetes") == 0:
            explanation.append("• diabetes: Well-controlled diabetes helps mitigate cardiovascular risks.")
        if input_data.get("Dyslipidemia") == 0:
            explanation.append("• dyslipidemia: The absence of dyslipidemia supports healthy cholesterol levels, reducing CAD risk.")
        if input_data.get("Chronic Kidney Disease") == 0:
            explanation.append("• chronic kidney disease: Absence of chronic kidney disease reduces the burden on the cardiovascular system.")
        if input_data.get("Asymptomatic") == 1:
            explanation.append("• asymptomatic: The absence of symptoms such as chest pain or shortness of breath can indicate a lower risk of severe cardiovascular disease.")
        if input_data.get("Atypical Symptoms") == 0:
            explanation.append("• atypical symptoms: The absence of atypical symptoms further supports a lower risk for CAD.")
        if input_data.get("ECG") == 1:
            explanation.append("• ECG: Normal ECG results suggest stable cardiac health without signs of acute heart disease.")
        if input_data.get("previous Stroke") == 0:
            explanation.append("• previous stroke: The absence of a history of stroke is associated with lower cardiovascular risk.")
        if input_data.get("previous AMI") == 0:
            explanation.append("• previous AMI: The absence of a history of myocardial infarction reduces the likelihood of CAD.")
        if input_data.get("previous PCI") == 0:
            explanation.append("• previous PCI: The absence of previous coronary interventions suggests stable coronary health.")
        if input_data.get("previous CABG") == 0:
            explanation.append("• previous CABG: Not having undergone coronary artery bypass grafting suggests less advanced coronary disease.")
        if input_data.get("Angina Like") == 0:
            explanation.append("• angina-like symptoms: The absence of angina-like symptoms further supports a lower likelihood of CAD.")
        if input_data.get("Dyspnoea on Exertion") == 0:
            explanation.append("• dyspnoea on exertion: The absence of shortness of breath during physical activity further supports a lower likelihood of CAD.")
        if input_data.get("Incident of Precordial Pain") == 0:
            explanation.append("• precordial pain: The absence of chest pain, particularly related to the heart, lowers the risk of CAD.")

                # Add more conditions as necessary.
        
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

#Read excel dataset
dataset=pd.read_excel("dataset.xlsx", engine='openpyxl')
initial_dataset = dataset.copy()

num_dimensions= dataset.shape[1] 

#Fill missing values with the method bfill
dataset.fillna(method="bfill", inplace=True)
column_names = dataset.columns[:-1]

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
                print(f"Column {column} has only one unique value, normalization skipped.")
        else:
            print(f"Column {column} is not numeric, skipping normalization.")



best_position = pd.read_excel("mean_values.xlsx", header=None)

best_position = best_position.to_numpy()[1:, :]

user_inputs_dict = {}


# If multiple rows, process each row
for index, row in dataset.iterrows():
    
    # Prepare the input for each row (second_row)
    user_inputs = row.tolist()  # Convert row to a list
    user_inputs_dict['SEX'] = initial_dataset['SEX'].iloc[index]
    user_inputs_dict['AGE'] = initial_dataset['AGE'].iloc[index]
    user_inputs_dict['BMI'] = initial_dataset['BMI'].iloc[index]

    num_dimensions, user_inputs_dict, best_position, user_inputs = remove_indices(user_inputs_dict, best_position, user_inputs )
    

    num_dimensions = num_dimensions -1
    
    # Apply the custom_prediction function
    output_prediction, predicted_results = custom_prediction(num_dimensions=num_dimensions, 
                                                                best_position=best_position, 
                                                                user_inputs=user_inputs)
    
    # Print the result for each instance (each row in the dataset)
    print(f"\n***************\nPatient {index + 1}:")

    if output_prediction > 0.85:
        classification = "pathological"
        print("The patient is likely to have CAD")
    else:
        classification = "normal"
        print("The patient is not likely to have CAD")
        
    plot_FCM_weight_matrix_graph(column_names=column_names, index=index,
                                    best_position=predicted_results)
    dict_results = dict(zip(list(column_names), predicted_results))

    interconnection_strength_text = {key: convert_to_strength(value) for key, value in dict_results.items()}
    input_data = dict(zip(list(column_names), user_inputs))
    
    input_data['SEX'] = user_inputs_dict['SEX']
    input_data['AGE'] = user_inputs_dict['AGE']
    input_data['BMI'] = user_inputs_dict['BMI']

    
    factual_explanation = generate_factual_explanation(input_data, interconnection_strength_text, classification)
  

    # Output the explanations
    print("\nFactual Explanation:")
    print(factual_explanation)
