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
    print(f"Total number of edges in the DAG: {dag_matrix_example.sum().sum()}")
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
    plt.title('Node DAG visualized')
    plt.axis('off')
    plt.show()
#=======================================================================================
# import networkx as nx
# import pandas as  pd
""" OBSOLETED
def generate_dataset_by_dag(dag_matrix_df, node_names, num_samples):
    Generates a synthetic dataset based on a given DAG structure.
    Args:
        dag_matrix (numpy.ndarray): An n x n adjacency matrix representing the DAG structure.
        node_names (list): A list of strings representing the names of the nodes.
        num_samples (int): The number of samples to generate.

    Returns:
        pandas.DataFrame: A DataFrame containing the generated dataset with columns corresponding to node names.
    # TODO check the validity of the DAG (e.g., acyclicity) and the consistency of node_names with dag_matrix dimensions.
    # Initialize an empty Pandas DataFrame for synthetic data
    data = pd.DataFrame(index=range(num_samples), columns=node_names)
    # TODO simplify, avoiding: dag_matrix_df = pd.DataFrame(dag_matrix_example, index=vars_example, columns=vars_example)
    # Coefficients for linear relationships
    # Using a range to ensure coefficients are not zero and vary.
    n = len(node_names)
    coeffs = np.random.uniform(0.5,2, size=(n,n)) * np.random.choice([-1, 1], size=(n, n))

    # Iterate through nodes in topological order (columns of true_dag_matrix_df)
    for i, node_c_name in enumerate(node_names):
        # Generate non-Gaussian noise (e.g., uniform distribution)
        # Centered uniform noise for non-Gaussianity
        noise = np.random.uniform(-1, 1, num_samples)
        # Initialize current node's value with noise
        current_node_value = noise
        # Get parents of the current node from the true_dag_matrix_df
        # true_dag_matrix_df has parents as rows (index) and children as columns
        # So, we look at the column of the current node to find its parents (rows with 1)
        parent_indices = np.where(dag_matrix_df.loc[:, node_c_name] == 1)[0]

        # Add contributions from parents
        for parent_idx in parent_indices:
            parent_name = node_names[parent_idx]
            # Ensure parent's data is already generated
            if data[parent_name].isnull().any():
            # This should not happen if processing in topological order,
            # but as a safeguard, skip if parent data is not ready.
            # In a true topological sort, parents are always processed before children.
                print(f"Warning: Parent {parent_name} data not fully generated for child {node_c_name}")
                continue

            # Get the coefficient for the edge from parent to current_node
            # We need the coefficient from parent_idx to i
            coefficient = coeffs[parent_idx, i]
            current_node_value += data[parent_name] * coefficient

        # Assign the calculated values to the current node's column in the DataFrame
        data[node_c_name] = current_node_value
    return data"""

#=======================================================================================
def generate_dataset_by_dag(dag_matrix_df, node_names, num_samples):
    # Create a directed graph from the adjacency matrix
    G_true = nx.from_pandas_adjacency(dag_matrix_df, create_using=nx.DiGraph)
    # Get a topological sort of the nodes
    topological_order = list(nx.topological_sort(G_true))

    # Initialize an empty Pandas DataFrame for synthetic data
    data = pd.DataFrame(index=range(num_samples), columns=topological_order)
    # Coefficients for linear relationships
    # Using a range to ensure coefficients are not zero and vary.
    num_to = len(topological_order)
    coeffs = np.random.uniform(0.5, 2, size=(num_to, num_to)) * np.random.choice(
        [-1, 1], size=(num_to, num_to))
    # Create a mapping from node name to its index in the topological order
    node_to_idx = {name: i for i, name in enumerate(topological_order)}

    # Iterate through nodes in the topological order
    for i, node_c_name in enumerate(topological_order):
        # Generate non-Gaussian noise (e.g., uniform distribution)
        # Centered uniform noise for non-Gaussianity
        noise = np.random.uniform(-1, 1, num_samples)

        # Initialize current node's value with noise
        current_node_value = noise

        # Get parents of the current node from the true_dag_matrix_df
        # true_dag_matrix_df has parents as rows (index) and children as columns
        # So, we look at the column of the current node to find its parents (rows with 1)
        # We need to use the original index for lookup in true_dag_matrix_df
        parent_original_indices = np.where(dag_matrix_df.loc[:, node_c_name] == 1)[0]

        # Add contributions from parents
        for parent_original_idx in parent_original_indices:
            parent_name = node_names[parent_original_idx]

            # Get the coefficient for the edge from parent to current_node
            # The `coeffs` matrix should be indexed by the order in `topological_order`
            # However, for simplicity and since `coeffs` is fully dense, we can use original indices
            # or re-map. For correctness and avoiding index errors, we should index `coeffs` by `node_to_idx`
            # Let's adjust coeffs to be indexed by original `fifty_node_names` for consistency with `true_dag_matrix_df`'s internal indexing.
            # Re-generating coeffs here to align with original `fifty_node_names` indices.
            # More robust: coeffs should be defined based on the original `fifty_node_names` indices, as it's a fixed size matrix.

            # For the coefficient, we need the original index of the parent and the original index of the child
            # The child's original index can be found using fifty_node_names.index(node_c_name)
            child_original_idx = node_names.index(node_c_name)
            coefficient = coeffs[parent_original_idx, child_original_idx]

            current_node_value += data[parent_name] * coefficient

        # Assign the calculated values to the current node's column in the DataFrame
        data[node_c_name] = current_node_value
    return data

if __name__ == "__main__":
    # Show the DAG adjacency matrix as a heatmap
    print("Generated synthetic observational data.")
    data_generated = generate_dataset_by_dag(dag_matrix_df, vars_example, num_samples=1000)
    print("First 5 rows of the generated data:")
    print(data_generated.head())

if __name__ == "__main__":
    # Plot the generated data pairvise to visualize relationships
    for i,j in itertools.combinations(vars_example, 2):
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=data_generated[i], y=data_generated[j], alpha=0.5)
        plt.title(f'Scatter plot of {i} vs {j}')
        plt.xlabel(i)
        plt.ylabel(j)
        plt.show()

#=======================================================================================
import lingam
import lingam.utils as lu

# Instantiate the DirectLiNGAM model
model = lingam.DirectLiNGAM()

# Fit the model to the data DataFrame
# The 'data' DataFrame was generated in the previous step and contains the synthetic observational data.
model.fit(data_generated)

# Extract the learned adjacency matrix
# The adjacency_matrix_ attribute returns a numpy array. Convert it to a Pandas DataFrame.
# Create a directed graph from the adjacency matrix
G_true = nx.from_pandas_adjacency(dag_matrix_df, create_using=nx.DiGraph)
# Get a topological sort of the nodes
topological_order = list(nx.topological_sort(G_true))

learned_dag_matrix_df = pd.DataFrame(model.adjacency_matrix_, index=topological_order, columns=topological_order)

print("Learned DAG adjacency matrix using DirectLiNGAM:")
print(learned_dag_matrix_df.head())
print(f"Dimensions of the learned DAG matrix: {learned_dag_matrix_df.shape}")
print(f"Total number of edges in the learned DAG: {learned_dag_matrix_df.sum().sum()}")

#=====================================================================================================
# Show the DAG adjacency matrix as a heatmap
# dag_matrix_df = pd.DataFrame(learned_dag_matrix_df, index=vars_example, columns=vars_example)

# Visualize the test DAG using NetworkX
# # Create a directed graph from the adjacency matrix
viz_graph = nx.from_numpy_array(learned_dag_matrix_df.values, create_using=nx.DiGraph)
# Map node indices to their names
viz_mapping = {i: name for i, name in enumerate(topological_order)}
G = nx.relabel_nodes(viz_graph, viz_mapping)

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=42)  # For consistent layout
nx.draw_networkx(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, arrowsize=20)
plt.title('Node DAG visualized')
plt.axis('off')
plt.show()