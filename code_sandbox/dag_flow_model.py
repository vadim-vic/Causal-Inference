import torch
import torch.nn as nn
import numpy as np

class RealNVP(nn.Module):
    def __init__(self, dim, hidden_dim, mask_config):
        super().__init__()
        self.dim = dim
        self.mask_config = mask_config # 0 for first half, 1 for second half

        self.s_net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2),
            nn.Tanh() # Ensure scale is bounded
        )
        self.t_net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2)
        )

    def forward(self, x):
        x1, x2 = (x[:, :self.dim // 2], x[:, self.dim // 2:]) if self.mask_config == 0 else \
                 (x[:, self.dim // 2:], x[:, :self.dim // 2])

        s = self.s_net(x1)
        t = self.t_net(x1)
        
        y2 = x2 * torch.exp(s) + t
        y1 = x1

        if self.mask_config == 0:
            y = torch.cat([y1, y2], dim=1)
        else:
            y = torch.cat([y2, y1], dim=1)
        
        log_det_J = torch.sum(s, dim=1)
        return y, log_det_J

    def inverse(self, y):
        y1, y2 = (y[:, :self.dim // 2], y[:, self.dim // 2:]) if self.mask_config == 0 else \
                 (y[:, self.dim // 2:], y[:, :self.dim // 2])

        s = self.s_net(y1)
        t = self.t_net(y1)
        
        x2 = (y2 - t) * torch.exp(-s)
        x1 = y1

        if self.mask_config == 0:
            x = torch.cat([x1, x2], dim=1)
        else:
            x = torch.cat([x2, x1], dim=1)
        
        return x


class DAGFlowModel(nn.Module):
    def __init__(self, n_nodes, hidden_dim=64, num_flow_layers=4):
        super().__init__()
        self.n_nodes = n_nodes
        self.adj_dim = n_nodes * n_nodes # Flattened adjacency matrix dimension
        self.latent_dim = hidden_dim # Latent dimension for the flow

        # Encoder: Maps flattened adjacency matrix to a latent space
        self.encoder = nn.Sequential(
            nn.Linear(self.adj_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_dim)
        )

        # Decoder: Maps latent space back to flattened adjacency matrix
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.adj_dim),
            nn.Sigmoid() # Output probabilities for each edge
        )

        # Normalizing Flow layers (RealNVP)
        self.flow_layers = nn.ModuleList()
        for i in range(num_flow_layers):
            # Alternate mask configurations
            mask_config = i % 2
            self.flow_layers.append(RealNVP(self.latent_dim, hidden_dim, mask_config))

    def forward(self, adj_matrix):
        # Flatten the adjacency matrix
        x = adj_matrix.view(-1, self.adj_dim)

        # Encode to latent space
        z = self.encoder(x)

        # Pass through normalizing flow
        log_det_J_total = 0
        for flow_layer in self.flow_layers:
            z, log_det_J = flow_layer(z)
            log_det_J_total += log_det_J

        # Decode from latent space
        output_adj_flat = self.decoder(z)

        # Reshape back to adjacency matrix
        output_adj_matrix = output_adj_flat.view(-1, self.n_nodes, self.n_nodes)

        return output_adj_matrix, log_det_J_total

    def sample(self, num_samples):
        # Sample from the base distribution (e.g., a standard Gaussian)
        z = torch.randn(num_samples, self.latent_dim)
        
        # Pass through the inverse flow
        for flow_layer in reversed(self.flow_layers):
            z = flow_layer.inverse(z)
        
        # Decode from latent space
        generated_adj_flat = self.decoder(z)
        generated_adj_matrix = generated_adj_flat.view(-1, self.n_nodes, self.n_nodes)
        
        # Apply threshold to get binary adjacency matrix
        generated_adj_matrix = (generated_adj_matrix > 0.5).float()

        # TODO: Add a mechanism to ensure acyclicity and no self-loops
        return generated_adj_matrix

    def log_prob(self, adj_matrix):
        # Flatten the adjacency matrix
        x = adj_matrix.view(-1, self.adj_dim)

        # Encode to latent space
        z = self.encoder(x)

        # Pass through normalizing flow and accumulate log_det_J
        log_det_J_total = 0
        for flow_layer in self.flow_layers:
            z, log_det_J = flow_layer(z)
            log_det_J_total += log_det_J
        
        # Log probability of the base distribution (standard Gaussian)
        log_prob_base = -0.5 * torch.sum(z**2 + torch.log(torch.tensor(2 * np.pi)), dim=1)
        
        return log_prob_base + log_det_J_total

    def ensure_dag_properties(self, adj_matrix_batch):
        """
        Ensures that the generated adjacency matrices are valid DAGs.
        This is a critical step and will likely involve:
        1. Removing self-loops.
        2. Enforcing acyclicity (e.g., using topological sort or a cycle detection algorithm
           and then removing edges to break cycles).
        """
        processed_dags = []
        for adj_matrix in adj_matrix_batch:
            # Remove self-loops
            np.fill_diagonal(adj_matrix, 0)

            # TODO: Implement acyclicity enforcement.
            # This is non-trivial for a batch and might require iterative edge removal
            # or a differentiable approximation if used within a neural network.
            # For now, we'll just return the matrices with self-loops removed.
            processed_dags.append(adj_matrix)
        return torch.tensor(np.array(processed_dags), dtype=torch.float32)
