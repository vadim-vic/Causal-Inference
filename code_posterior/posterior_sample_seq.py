import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
#===========================
# This code trains a simple feedforward neural network on synthetic data and collects the weights and biases of each linear layer after each epoch. The parameters are stored in a dictionary where the keys are the module names and the values are 3D numpy arrays containing the parameter samples across epochs. After training, it plots histograms of the collected parameter samples for each weight and bias.
import numpy as np
import matplotlib.pyplot as plt
#===========================
from sklearn.datasets import load_iris
#===========================
#%%

# Reproducibility
torch.manual_seed(0)

# 1. Generate synthetic data
#N = 2000
#Z = torch.randn(N, 1)
#X = 2 * Z + 0.5 * torch.randn(N, 1)
#Y = 3 * X + 4 * Z + torch.randn(N, 1)

# 2. Create dataset and dataloader
#dataset = TensorDataset(X, Z, Y)
#dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 1. Dataset: 100 samples, 8 features
X = torch.randn(1000, 8)
y = torch.randint(0, 2, (1000,))

# 2. Create dataloader
loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)

# 3. Define a simple feedforward neural network
model = nn.Sequential(
    nn.Linear(8, 4),   # 4 neurons
    nn.ReLU(),
    nn.Linear(4, 2)    # 2 neurons
)

module_parameter_samples = {}  # Dictionary to store parameter samples for each module
def parameter_sample_set(model, module_parameter_samples):
# Gather each layer as a matrix of weights and biases (the last column is bias)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"\nLayer: {name} ({module.in_features} → {module.out_features})")

            W = module.weight.detach().numpy()  # [out_features, in_features]
            b = module.bias.detach().numpy() if module.bias is not None else None
            # Append bias to weights
            if b is not None:
                Wb = np.hstack([W, b.reshape(-1, 1)])
            else:
                Wb = W
            # print(f"\n{Wb}")
            # Initialize list if not exists
            if name not in module_parameter_samples:
                module_parameter_samples[name] = Wb[:, :, np.newaxis]  # Start with the first sample
            else:
                module_parameter_samples[name] =  np.concatenate((module_parameter_samples[name], Wb[:, :, np.newaxis]), axis=2)
            # Append the current Wb to the list of samples for this module
            # by the third dimension (stacking matrices)
            # module_parameter_samples[name].append(Wb)
    return module_parameter_samples

def print_linear_params_by_neuron(model, epoch):
    print(f"\n===== Epoch {epoch} =====")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"\nLayer: {name} ({module.in_features} → {module.out_features})")

            W = module.weight.detach().numpy()  # [out_features, in_features]
            b = module.bias.detach().numpy() if module.bias is not None else None
            # Append bias to weights
            if b is not None:
                Wb = np.hstack([W, b.reshape(-1, 1)])
            else:
                Wb = W
            # print(f"\n{Wb}")
            # print(Wb.size())

            #for i in range(W.size(0)):  # out_features = neurons
            #    print(f"  Neuron {i}:")
            #    print(f"    weights: {W[i].tolist()}")
            #    if b is not None:
            #        print(f"    bias: {b[i].item()}")

            plt.imshow(Wb, cmap='viridis', aspect='auto')
            plt.show()
    return Wb

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

num_epochs = 34

## Training 1: Observed dataset, no interventions
for epoch in range(num_epochs):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        module_parameter_samples = parameter_sample_set(model, module_parameter_samples)

    # Print neuron parameters AFTER the epoch update
    Wb = print_linear_params_by_neuron(model, epoch)
    module_parameter_samples = parameter_sample_set(model, module_parameter_samples)
    print(Wb.shape)  # Should be [out_features, in_features + 1] for the last linear layer

## Training 2: Intervention into the dataset
# Pick a random feature and set it to zero for all samples
intervention_feature = np.random.randint(0, X.shape[1])
print(f"\nIntervening on feature {intervention_feature} (setting it to zero)")
X[:, intervention_feature] = 0.0  # Set the chosen feature to zero for all samples
# Create a new dataloader with the intervened dataset
intervened_loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
for epoch in range(num_epochs):
    model.train()
    for xb, yb in intervened_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        module_parameter_samples_inter = parameter_sample_set(model, module_parameter_samples)

    # Print neuron parameters AFTER the epoch update
    Wb = print_linear_params_by_neuron(model, epoch)
    module_parameter_samples = parameter_sample_set(model, module_parameter_samples)
    print(Wb.shape)  # Should be [out_features, in_features + 1] for the last linear layer


for name, samples in module_parameter_samples.items():
    print(f"\nModule: {name}, Samples shape: {samples.shape}")



for name, samples in module_parameter_samples.items():
    for i in range(samples.shape[0]):
        for j in range(samples.shape[1]):
            x = samples[i, j, :]
            # plot histogram of the samples for this weight/bias
            plt.hist(x, bins=10, alpha=0.5)
            plt.title(f"Module: {name}, Weight[{i},{j}] samples: {x.shape[0]}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.show()


