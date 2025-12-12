import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------
# 1. Simulate Confounded Data
# --------------------------
N = 5000

torch.manual_seed(0)
Z = torch.randn(N, 1)                      # confounder
X = 0.8 * Z + torch.randn(N, 1)            # Z -> X
Y = 1.5 * X + 2 * Z + torch.randn(N, 1)    # X -> Y and Z -> Y

# --------------------------
# Simple feed-forward model
# --------------------------
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# --------------------------
# 2. Observational model P(Y|X)
# --------------------------
model_obs = MLP(input_dim=1)
opt = optim.Adam(model_obs.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for _ in range(1000):
    opt.zero_grad()
    pred = model_obs(X)
    loss = loss_fn(pred, Y)
    loss.backward()
    opt.step()

# Estimate causal effect as slope via local derivative
X_test = torch.linspace(-2, 2, 100).unsqueeze(1)
Y_hat = model_obs(X_test)
slope_obs = torch.autograd.functional.jacobian(
    lambda x: model_obs(x).mean(), X_test
).mean().item()

print("Observational slope (biased):", slope_obs)

# --------------------------
# 3. Causal model P(Y|X,Z)
# --------------------------
XZ = torch.cat([X, Z], dim=1)
model_causal = MLP(input_dim=2)
opt = optim.Adam(model_causal.parameters(), lr=0.01)

for _ in range(1000):
    opt.zero_grad()
    pred = model_causal(XZ)
    loss = loss_fn(pred, Y)
    loss.backward()
    opt.step()
'''
# Estimate causal effect: derivative wrt X while holding Z fixed
def causal_fn(x):
    # x is a tensor of [X, Z]
    return model_causal(x).mean()

# Build test input: vary X, hold Z = 0
XZ_test = torch.cat([X_test, torch.zeros_like(X_test)], dim=1)

'''
# Define a function that accepts a *single input tensor*
def causal_fn(x):
    return model_causal(x)

# Compute the Jacobian wrt input
'''J = torch.autograd.functional.jacobian(causal_fn, X_test)[:, :, 0].mean().item()  # partial derivative wrt X

# J has shape: [N, 1, N, 2]
# We want the derivative of Y wrt X *for each row*
# That corresponds to J[i, 0, i, 0]
slopes = []
for i in range(len(X_test)):
    slopes.append(J[i, 0, i, 0].item())

slope_causal = sum(slopes) / len(slopes)
print("Causal slope (adjusted):", slope_causal)
'''



XZ_test = torch.cat([X_test, torch.zeros_like(X_test)], dim=1)
slope_causal = torch.autograd.functional.jacobian(causal_fn, XZ_test)[:, :, 0].mean().item()  # partial derivative wrt X

print("Causal slope (adjusted):", slope_causal)
