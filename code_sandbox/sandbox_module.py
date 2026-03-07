"""
def generate_variable_names(num_vars): -> generated_vars
def generate_random_dag_matrix(node_names, sparsity): -> dag_df


"""

import string
import itertools

def generate_variable_names(num_vars):
    """
    Generates a set of unique one, two, or three-letter variable names.

    Args:
        num_vars (int): The desired number of variable names to generate.

    Returns:
        set: A set containing the generated variable names.

    Example usage:
        num_vars_example = 345 # Using the original number of variables
        vars_example = generate_variable_names(num_vars_example)
        print(f"Generated {len(vars_example)} variables: {vars_example}")
    """

    if num_vars > 17630:
        raise ValueError("Cannot generate more than 17,630 unique variable names with 1-3 letters.")

    generated_vars = list()
    alphabet = string.ascii_uppercase

    # Generate 1-letter variables
    for char in alphabet:
        if len(generated_vars) < num_vars:
            generated_vars.append(char)
        else:
            break

    # Generate 2-letter variables
    if len(generated_vars) < num_vars:
        for pair in itertools.product(alphabet, repeat=2):
            if len(generated_vars) < num_vars:
                generated_vars.append("".join(pair))
            else:
                break

    # Generate 3-letter variables
    if len(generated_vars) < num_vars:
        for triplet in itertools.product(alphabet, repeat=3):
            if len(generated_vars) < num_vars:
                generated_vars.append("".join(triplet))
            else:
                break
    return generated_vars

if __name__ == "__main__": # Stub for tests
    # Minimum example
    vars_show = ['Z','Y','X','U','W']
    print(f"Asserted {len(vars_show)} variables: {vars_show}")
    # Example usage:
    num_vars_example = 5 # Using the original number of variables
    vars_example = generate_variable_names(num_vars_example)
    print(f"Generated {len(vars_example)} variables: {vars_example}")

#=======================================================================================
import numpy as np
import random

def generate_random_dag_matrix(node_names, sparsity):
    """
    Generates a random Directed Acyclic Graph (DAG) adjacency matrix
    with a given sparsity.
    # TODO: There is no need in the node_names argument, only the number of variables.

    Args:
        node_names (list): A list of strings representing the names of the nodes.
        sparsity (float): A value between 0 and 1 indicating the desired density
                          of edges. Higher sparsity means more edges.

    Returns:
        pandas.DataFrame: An n x n adjacency matrix where n is the number of nodes.
                          (obsoleted) The matrix is a Pandas DataFrame with node names as
                          both index and columns.
    """
    if not (0 <= sparsity <= 1):
        raise ValueError("Sparsity must be a value between 0 and 1.")
    if node_names is None:
        raise ValueError("Node names cannot be None.")
    if len(node_names) == 0:
        raise ValueError("Node names list cannot be empty.")

    n = len(node_names)
    adj_matrix = np.zeros((n, n), dtype=int)

    # Create a random topological ordering of the node indices
    random_order = list(range(n))
    random.shuffle(random_order)

    # Iterate through all possible ordered pairs of nodes (u, v)
    # such that u appears before v in the random topological ordering
    for i in range(n):
        u_original_idx = random_order[i]
        for j in range(i + 1, n):
            v_original_idx = random_order[j]

            # Generate a random number. If less than sparsity, add an edge.
            if random.random() < sparsity:
                adj_matrix[u_original_idx, v_original_idx] = 1

    # (obsoleted) Convert the NumPy matrix to a Pandas DataFrame
    # dag_df = pd.DataFrame(adj_matrix, index=node_names, columns=node_names)
    return adj_matrix

if __name__ == "__main__": # Stub for tests
    # Minimum example for var_show = ['Z','Y','X','U','W']
    dag_show = [[0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0]]
    print(f"Asserted DAG adjacency matrix:\n{dag_show}")
    # Example usage:
    assign_sparsity=0.5 # Density (0.0 = no edges, 1.0 = all possible edges)
    dag_matrix_example = generate_random_dag_matrix(
        vars_example, #generate_variable_names(34),
        assign_sparsity)
    print(f"Generated DAG adjacency matrix with shape {dag_matrix_example.shape} and sparsity {assign_sparsity}:\n{dag_matrix_example}")

#=======================================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

if __name__ == "__main__":
    # Show the DAG adjacency matrix as a heatmap
    dag_matrix_df  = pd.DataFrame(dag_matrix_example, index=vars_example, columns=vars_example)
    # Visualize the test DAG using heat matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(dag_matrix_df, cmap='viridis', cbar=False)
    plt.title('Test DAG Adjacency Matrix')
    plt.xlabel('Destination Node')
    plt.ylabel('Source Node')
    plt.show()

    # Visualize the test DAG using NetworkX
    # # Create a directed graph from the adjacency matrix
    viz_graph = nx.from_numpy_array(dag_matrix_example, create_using=nx.DiGraph)
    # Map node indices to their names
    viz_mapping = {i: name for i, name in enumerate(vars_example)}
    G = nx.relabel_nodes(viz_graph, viz_mapping)

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42) # For consistent layout
    nx.draw_networkx(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, arrowsize=20)
    plt.title('5-Node DAG visualized with NetworkX')
    plt.axis('off')
    plt.show()

