import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# 1. Whitening (PCA)
# ----------------------------
def whiten(X, eps=1e-5):
    """
    X: (batch, dim)
    Returns:
        X_white: whitened data
        whitening_matrix
    """
    X = X - X.mean(dim=0, keepdim=True)

    cov = X.T @ X / X.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # Numerical stability
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(eigvals + eps))

    whitening_matrix = eigvecs @ D_inv_sqrt @ eigvecs.T
    X_white = X @ whitening_matrix

    return X_white, whitening_matrix


# ----------------------------
# 2. ICA Model
# ----------------------------
class ICAModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)

        # Initialize close to orthogonal
        nn.init.orthogonal_(self.W.weight)

    def forward(self, x):
        return self.W(x)


# ----------------------------
# 3. Loss functions
# ----------------------------

def negentropy_loss(z):
    """
    Approximate negentropy using log-cosh
    Encourages non-Gaussianity
    """
    return torch.mean(torch.log(torch.cosh(z)))


def orthogonality_loss(W):
    """
    Enforce W W^T ≈ I
    """
    I = torch.eye(W.size(0), device=W.device)
    WT_W = W @ W.T
    return torch.norm(WT_W - I)


# ----------------------------
# 4. Training loop
# ----------------------------
def train_ica(X, epochs=1000, lr=1e-3, lambda_ortho=1.0):
    """
    X: (N, dim)
    """
    device = X.device
    dim = X.shape[1]

    # Whitening
    X_white, whitening_matrix = whiten(X)

    model = ICAModel(dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        z = model(X_white)

        loss_indep = negentropy_loss(z)
        loss_ortho = orthogonality_loss(model.W.weight)

        loss = loss_indep + lambda_ortho * loss_ortho

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    return model, whitening_matrix


# ----------------------------
# 5. Recover sources
# ----------------------------
def recover_sources(model, whitening_matrix, X):
    X_centered = X - X.mean(dim=0, keepdim=True)
    X_white = X_centered @ whitening_matrix
    S = model(X_white)
    return S


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    N = 5000
    dim = 3

    # Create synthetic independent sources
    S_true = torch.stack([
        torch.sin(torch.linspace(0, 20, N)),
        torch.sign(torch.sin(torch.linspace(0, 10, N))),
        torch.randn(N)
    ], dim=1)

    # Mix them
    A = torch.randn(dim, dim)
    X = S_true @ A.T

    # Train ICA
    model, W_white = train_ica(X, epochs=1000)

    # Recover signals
    S_est = recover_sources(model, W_white, X)

    print("Done!")