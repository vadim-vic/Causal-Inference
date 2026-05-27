import torch
import torch.optim as optim
from dag_flow_model import DAGFlowModel
from causal_dataset_generator import CausalDatasetGenerator
import numpy as np
import networkx as nx
from tqdm import tqdm

def generate_random_dag_adj(n_nodes, p_edge):
    """Generates a random DAG adjacency matrix."""
    adj = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes): # Ensure acyclicity by only allowing i -> j where i < j
            if np.random.rand() < p_edge:
                adj[i, j] = 1
    return adj

def is_acyclic(adj_matrix):
    """Checks if an adjacency matrix represents a DAG."""
    graph = nx.DiGraph(adj_matrix)
    return nx.is_directed_acyclic_graph(graph)

def remove_cycles(adj_matrix):
    """
    Removes edges from an adjacency matrix to ensure it's a DAG.
    This is a greedy approach and might not be optimal.
    """
    adj_copy = adj_matrix.copy()
    graph = nx.DiGraph(adj_copy)
    
    while not nx.is_directed_acyclic_graph(graph):
        try:
            cycle = nx.find_cycle(graph)
            # Remove one edge from the cycle. For simplicity, remove the last edge found.
            u, v, _ = cycle[-1]
            adj_copy[u, v] = 0
            graph = nx.DiGraph(adj_copy)
        except nx.NetworkXNoCycle:
            break # No cycle found, should not happen if loop condition is correct
    return adj_copy

def main():
    # Hyperparameters
    n_nodes = 5
    p_edge = 0.3
    num_dags_train = 1000
    num_dags_test = 100
    hidden_dim = 64
    num_flow_layers = 4
    learning_rate = 1e-3
    num_epochs = 100
    batch_size = 32

    # 1. Generate synthetic DAGs
    print(f"Generating {num_dags_train} training DAGs...")
    train_dags = []
    for _ in tqdm(range(num_dags_train)):
        adj = generate_random_dag_adj(n_nodes, p_edge)
        train_dags.append(adj)
    train_dags = torch.tensor(np.array(train_dags), dtype=torch.float32)

    print(f"Generating {num_dags_test} test DAGs...")
    test_dags = []
    for _ in tqdm(range(num_dags_test)):
        adj = generate_random_dag_adj(n_nodes, p_edge)
        test_dags.append(adj)
    test_dags = torch.tensor(np.array(test_dags), dtype=torch.float32)

    # 2. Initialize model, optimizer
    model = DAGFlowModel(n_nodes, hidden_dim, num_flow_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i in range(0, num_dags_train, batch_size):
            batch_adj = train_dags[i:i+batch_size]
            
            optimizer.zero_grad()
            log_prob = model.log_prob(batch_adj)
            loss = -torch.mean(log_prob) # Negative log-likelihood
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / (num_dags_train / batch_size):.4f}")

    # 4. Evaluation
    print("\nEvaluating model...")
    model.eval()
    with torch.no_grad():
        # Calculate average log-likelihood on test set
        test_log_prob = model.log_prob(test_dags)
        avg_test_nll = -torch.mean(test_log_prob).item()
        print(f"Average Test Negative Log-Likelihood: {avg_test_nll:.4f}")

        # Generate new DAGs and check validity
        num_generated_dags = 100
        generated_dags_raw = model.sample(num_generated_dags)
        
        # Apply post-processing to ensure DAG properties
        valid_generated_dags = []
        for i in range(num_generated_dags):
            processed_adj = remove_cycles(generated_dags_raw[i].cpu().numpy())
            if is_acyclic(processed_adj):
                valid_generated_dags.append(processed_adj)
        
        validity_ratio = len(valid_generated_dags) / num_generated_dags
        print(f"Generated {num_generated_dags} DAGs. {len(valid_generated_dags)} were valid after post-processing.")
        print(f"Validity Ratio: {validity_ratio:.2f}")

        # Optional: Print some generated DAGs
        print("\nExample Generated DAGs (after post-processing):")
        for i, dag in enumerate(valid_generated_dags[:5]):
            print(f"DAG {i+1}:\n{dag.astype(int)}")

if __name__ == "__main__":
    main()
