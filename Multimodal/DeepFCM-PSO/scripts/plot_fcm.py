import networkx as nx
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

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



def plot_FCM_weight_matrix_graph(column_names, best_position, scale_factor=2.5):
    # Determine the output node based on the study
    output_node = 'Output'

    # Create directed graph
    G = nx.DiGraph()
    # st.write(column_names, best_position)
    # Add nodes to the graph
    G.add_nodes_from(column_names + [output_node])

    # Round the best_position weights to two decimals
    best_position = [round(weight, 2) for weight in best_position]

    # Add weighted edges to the graph based on the best_position list
    for i, weight in enumerate(best_position):
        if abs(weight) > 0.05:  # Lower the threshold to include more edges
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

    # Show the plot
    plt.savefig('FCM_weight_matrix_graph.png', format='png', dpi=300, bbox_inches='tight')
