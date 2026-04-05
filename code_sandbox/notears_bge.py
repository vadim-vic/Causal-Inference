"""
NOTEARS with Bayesian Gaussian equivalent (BGe) score
======================================================
Continuous optimisation for sparse DAG structure learning.

Reference: Zheng et al. (2018) "DAGs with NO TEARS"
           Geiger & Heckerman (2002) BGe score

Usage
-----
    python notears_bge.py          # runs a synthetic demo
"""

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# BGe local score and gradient
# ---------------------------------------------------------------------------

def bge_local_score(xi: np.ndarray, Xpa: np.ndarray,
                    lam: float = 1.0,
                    alpha0: float = 1.0, beta0: float = 1.0) -> float:
    """
    Log marginal likelihood of node i given its parents (BGe score).

    Parameters
    ----------
    xi    : (m,)   observations of node i
    Xpa   : (m, k) observations of parent nodes  (k=0 => no parents)
    lam   : prior precision on edge weights W_ij ~ N(0, 1/lam)
    alpha0, beta0 : Inv-Gamma prior hyperparameters on noise variance

    Returns
    -------
    scalar log p(X_i | X_pa(i))
    """
    m = len(xi)
    k = Xpa.shape[1] if Xpa.ndim == 2 else 0

    if k == 0:
        # No parents: residual is xi itself
        rss = xi @ xi
    else:
        A = Xpa.T @ Xpa + lam * np.eye(k)       # (k x k) regularised gram
        b = Xpa.T @ xi                            # (k,)
        coef = np.linalg.solve(A, b)             # ridge estimate
        residual = xi - Xpa @ coef
        rss = residual @ residual

    beta_hat = beta0 + 0.5 * rss

    score = (
        gammaln(alpha0 + m / 2)
        - gammaln(alpha0)
        + alpha0 * np.log(beta0)
        - (alpha0 + m / 2) * np.log(beta_hat)
        - 0.5 * m * np.log(2 * np.pi)
        - 0.5 * np.log(lam)          # -0.5*(log lam^(k+1) - log lam^k)
    )
    return float(score)


def bge_local_grad(xi: np.ndarray, Xpa: np.ndarray,
                   lam: float = 1.0,
                   alpha0: float = 1.0, beta0: float = 1.0) -> np.ndarray:
    """
    Gradient of BGe local score w.r.t. the edge weights W_i (row i of W).

    The weights enter through Xpa = X @ W_i^T (columns selected by pa(i)).
    Here we return the gradient w.r.t. the *full* row W[i, :] of shape (n,),
    so entries corresponding to non-parents are zero.

    Returns
    -------
    grad : (k,) gradient w.r.t. the k parent edge weights
    """
    m = len(xi)
    k = Xpa.shape[1] if Xpa.ndim == 2 else 0
    if k == 0:
        return np.zeros(0)

    A = Xpa.T @ Xpa + lam * np.eye(k)
    b = Xpa.T @ xi
    coef = np.linalg.solve(A, b)
    residual = xi - Xpa @ coef
    rss = residual @ residual
    beta_hat = beta0 + 0.5 * rss

    # d(beta_hat)/d(coef) = -X_pa^T residual  (via chain rule on rss)
    # d(score)/d(beta_hat) = -(alpha0 + m/2) / beta_hat
    # d(coef)/d(W_i[j]) = A^{-1} X_pa[:,j]^T x_i  -- but W enters linearly
    # Simpler: d(score)/d(W_i) = d(score)/d(beta_hat) * d(beta_hat)/d(W_i)
    #   d(beta_hat)/d(W_i) = -X_pa^T residual (chain rule)
    factor = -(alpha0 + m / 2) / beta_hat   # scalar
    # d(rss)/d(coef) and d(coef)/d(W_i) are via the normal equations
    # final: grad w.r.t. parent coefficients
    grad_coef = factor * (-Xpa.T @ residual)   # (k,)
    return grad_coef


# ---------------------------------------------------------------------------
# Acyclicity constraint  h(W) = tr(e^{W circ W}) - n
# ---------------------------------------------------------------------------

def acyclicity(W: np.ndarray):
    """h(W) = tr(exp(W o W)) - n"""
    WW = W * W
    return np.trace(expm(WW)) - W.shape[0]


def acyclicity_grad(W: np.ndarray) -> np.ndarray:
    """∇_W h(W) = 2 W o exp(W o W)"""
    WW = W * W
    return 2.0 * W * expm(WW)


# ---------------------------------------------------------------------------
# Full augmented Lagrangian objective
# ---------------------------------------------------------------------------

class NOTEARS_BGe:
    """
    NOTEARS with BGe score + L1 sparsity via augmented Lagrangian.

    Parameters
    ----------
    lam_prior : ridge prior precision on edge weights
    alpha0, beta0 : BGe Inv-Gamma hyperparameters
    gamma  : L1 sparsity coefficient
    rho0   : initial penalty coefficient
    mu     : penalty growth factor
    delta  : acyclicity tolerance for convergence
    max_outer : maximum outer (AL) iterations
    """

    def __init__(self, lam_prior=1.0, alpha0=1.0, beta0=1.0,
                 gamma=0.1, rho0=1.0, mu=10.0, delta=1e-8, max_outer=20):
        self.lam_prior = lam_prior
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.gamma = gamma
        self.rho0 = rho0
        self.mu = mu
        self.delta = delta
        self.max_outer = max_outer

    def fit(self, X: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Learn a sparse DAG from data X.

        Parameters
        ----------
        X : (m, n)  data matrix, m samples, n variables

        Returns
        -------
        W : (n, n)  estimated weighted adjacency matrix (DAG)
        """
        m, n = X.shape
        W = np.zeros((n, n))
        alpha = 0.0
        rho = self.rho0

        def objective_and_grad(w_flat):
            W = w_flat.reshape(n, n)
            np.fill_diagonal(W, 0.0)          # no self-loops

            # --- negative BGe score ---
            loss = 0.0
            grad_W = np.zeros((n, n))

            for i in range(n):
                pa_idx = np.where(np.abs(W[:, i]) > 0)[0]  # parents of i
                xi = X[:, i]

                if len(pa_idx) == 0:
                    loss -= bge_local_score(xi, np.empty((m, 0)),
                                            self.lam_prior, self.alpha0, self.beta0)
                else:
                    Xpa = X[:, pa_idx]
                    loss -= bge_local_score(xi, Xpa,
                                            self.lam_prior, self.alpha0, self.beta0)
                    g = bge_local_grad(xi, Xpa,
                                       self.lam_prior, self.alpha0, self.beta0)
                    grad_W[pa_idx, i] -= g

            # --- L1 sparsity (subgradient) ---
            loss += self.gamma * np.sum(np.abs(W))
            grad_W += self.gamma * np.sign(W)

            # --- acyclicity ---
            h = acyclicity(W)
            grad_h = acyclicity_grad(W)
            loss += alpha * h + 0.5 * rho * h ** 2
            grad_W += (alpha + rho * h) * grad_h

            np.fill_diagonal(grad_W, 0.0)
            return float(loss), grad_W.flatten()

        for t in range(self.max_outer):
            # Inner minimisation (L-BFGS-B)
            result = minimize(
                objective_and_grad,
                W.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            W = result.x.reshape(n, n)
            np.fill_diagonal(W, 0.0)

            # Hard-threshold small weights (promote sparsity)
            W[np.abs(W) < 0.3] = 0.0

            h = acyclicity(W)
            if verbose:
                edges = int(np.sum(np.abs(W) > 0))
                print(f"  outer iter {t+1:2d} | h(W) = {h:.2e} | "
                      f"edges = {edges} | rho = {rho:.1e}")

            if h <= self.delta:
                if verbose:
                    print("  Converged: acyclicity satisfied.")
                break

            # Update multipliers
            alpha = alpha + rho * h
            rho = self.mu * rho

        return W


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def shd(W_true: np.ndarray, W_est: np.ndarray, thresh: float = 0.3) -> int:
    """Structural Hamming distance between two DAGs."""
    true_edges = set(zip(*np.where(np.abs(W_true) > thresh)))
    est_edges  = set(zip(*np.where(np.abs(W_est)  > thresh)))
    return len(true_edges.symmetric_difference(est_edges))


def simulate_dag(n: int, edge_prob: float = 0.3,
                 weight_range=(0.5, 2.0), seed: int = 0) -> np.ndarray:
    """Generate a random DAG adjacency matrix (lower-triangular)."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            if rng.random() < edge_prob:
                w = rng.uniform(*weight_range)
                W[i, j] = w * rng.choice([-1, 1])
    return W


def simulate_data(W: np.ndarray, m: int, sigma: float = 1.0,
                  seed: int = 1) -> np.ndarray:
    """
    Sample m observations from x = Wx + eps, eps ~ N(0, sigma^2 I).
    Solves: x = (I - W)^{-1} eps
    """
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    eps = rng.normal(0, sigma, size=(m, n))
    IminusW_inv = np.linalg.inv(np.eye(n) - W)
    return eps @ IminusW_inv.T


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  NOTEARS + BGe  —  sparse DAG structure learning")
    print("=" * 55)

    # Ground truth
    n, m = 6, 300
    W_true = simulate_dag(n, edge_prob=0.35, seed=42)
    X = simulate_data(W_true, m, sigma=1.0, seed=7)

    true_edges = int(np.sum(np.abs(W_true) > 0.3))
    print(f"\nGround truth: {n} nodes, {true_edges} edges")
    print("True W (lower-triangular):")
    print(np.round(W_true, 2))

    # Fit
    print("\nRunning NOTEARS-BGe optimisation...")
    model = NOTEARS_BGe(
        lam_prior=1.0, alpha0=1.0, beta0=1.0,
        gamma=0.05, rho0=1.0, mu=10.0, delta=1e-8, max_outer=25
    )
    W_est = model.fit(X, verbose=True)

    # Results
    est_edges = int(np.sum(np.abs(W_est) > 0.3))
    h_final = acyclicity(W_est)
    distance = shd(W_true, W_est)

    print(f"\nEstimated W:")
    print(np.round(W_est, 2))
    print(f"\nResults")
    print(f"  True edges     : {true_edges}")
    print(f"  Estimated edges: {est_edges}")
    print(f"  h(W) (acyclic) : {h_final:.2e}  (target ≈ 0)")
    print(f"  SHD            : {distance}  (lower = better)")
