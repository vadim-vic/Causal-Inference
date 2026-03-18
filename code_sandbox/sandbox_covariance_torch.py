import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define a two-layer neural network
class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
# Initialize model, loss, and optimizer
model = TwoLayerNN(input_size=10, hidden_size=64, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create dummy data
X = torch.randn(100, 10)
y = torch.randn(100, 1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training loop with mini-batch
num_epochs = 10
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
# %%
#%%
# Gather the model parameters for covariance calculation
def get_model_parameters(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    return torch.cat(params)
# Example usage
model = TwoLayerNN(input_size=10, hidden_size=64, output_size=1)
params = get_model_parameters(model)
print("Model parameters shape:", params.shape)  
#%%
# Calculate the covariance matrix of the model parameters
def calculate_covariance_matrix(params):
    mean = torch.mean(params)
    centered_params = params - mean
    covariance_matrix = torch.mm(centered_params.unsqueeze(1), centered_params.unsqueeze(0)) / (params.size(0) - 1)
    return covariance_matrix
# Example usage
cov_matrix = calculate_covariance_matrix(params)
print("Covariance matrix shape:", cov_matrix.shape) 