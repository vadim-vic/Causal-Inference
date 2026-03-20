# AGENTS.md - Causal Inference Codebase Guide

This guide helps AI agents understand and contribute effectively to the Causal Inference project.

## Project Overview

**Purpose**: Research and implementation of causal inference methods for time series and text data, with emphasis on parameter estimation and posterior distribution analysis.

**Structure**:
- `code_posterior/`: Bayesian parameter estimation and posterior distribution visualization (notebooks + Python scripts)
- `code_sandbox/`: Experimental causal inference implementations and case studies (notebooks + modules)
- `slides/`: Theoretical papers and presentation materials (LaTeX + PDF)

## Architecture Patterns

### 1. Structural Causal Models (SCMs) and DAGs

The project heavily uses Directed Acyclic Graphs (DAGs) to represent causal relationships.

**Key Class**: `CausalDatasetGenerator` (`code_sandbox/causal_dataset_generator.py`)
- Takes adjacency matrix representation of DAG
- Performs topological sort via `predecessors_traversal_order()` to determine causal order
- Builds symbolic structural equations via `build_structural_equations(nonlinear=True/False)`
- Generates synthetic datasets with `generate_dataset_from_dag(n_samples, nonlinear)`
- Supports both linear mechanisms (`w_i*X_i`) and nonlinear (`tanh()`, `sin()`, `sigmoid()`, `square()`)

**Example Usage Pattern**:
```python
# Define causal graph as adjacency matrix
adj = [[0,1,1,0], [0,0,0,1], [0,0,0,1], [0,0,0,0]]  # 0→1, 0→2, 1→3, 2→3
gen = CausalDatasetGenerator(adj)
df, equations = gen.generate_dataset_from_dag(n_samples=1000, nonlinear=False)
```

### 2. Parameter Estimation and Posterior Distributions

Core workflow in `code_posterior/`:
1. **Sample generation**: Train models on multiple dataset batches
2. **Parameter collection**: Gather weights/biases across epochs (3D tensors: `[epochs, layer, param]`)
3. **Density estimation**: Estimate probability distributions over collected parameters using:
   - Gaussian models
   - Flow-based models (normalizing flows)

**Pattern Example** (`code_posterior/posterior_sample_seq.py`):
```python
# Collect parameters across training epochs
module_parameter_samples = {}  # {module_name: 3D_numpy_array}
for epoch in range(num_epochs):
    model.train()
    # ... training loop ...
    module_parameter_samples = parameter_sample_set(model, module_parameter_samples)

# Analyze parameter distributions
for name, samples in module_parameter_samples.items():
    # samples.shape = [num_epochs, out_features, in_features + 1]
```

### 3. Causal Effect Estimation

Two main approaches implemented:

**a) Confounding-aware models** (`code_sandbox/causal_bayes_*.py`):
- Naive observational: `Y ~ X` (biased)
- Causal adjustment: `Y ~ X + Z` (unbiased if Z is all confounders)
- Uses `do(X=x)` operator notation for interventional distributions

**b) Interventional analysis** (`code_posterior/posterior_sample_seq.py`):
- Train on original data, collect parameters
- Intervene on features (set to zero), retrain
- Compare parameter distributions before/after intervention
- Shows how causal effects manifest in neural network weights

## Developer Workflows

### Running Experiments

**Notebooks vs Scripts**:
- **Notebooks** (`.ipynb`): Exploratory analysis, visualization, parameter tuning
- **Scripts** (`.py`): Reproducible pipelines, dataset generation, testing

**Testing Framework**:
- Unit tests: `code_sandbox/test_causal_dataset_generator.py` (uses `pytest`)
- Fixtures: Simple DAG structure repeated across tests
- Run tests: `pytest test_causal_dataset_generator.py`

**Typical Workflow**:
1. Prototype in notebook (e.g., `sandbox_notebook.ipynb`)
2. Extract reusable components to modules (e.g., `sandbox_module.py`)
3. Write tests for generator/data functions
4. Create reproducible script version

### Dependencies

No `requirements.txt`; inferred from code imports:
- **PyTorch**: Neural network models (`torch`, `torch.nn`, `torch.optim`)
- **scikit-learn**: Baselines (`LinearRegression`, `MLP`)
- **NumPy/Pandas**: Data manipulation
- **LiNGAM**: Causal discovery (`lingam`, `lingam.utils`)
- **Visualization**: `matplotlib`, `seaborn`, `networkx` (for graph plotting)
- **Time series**: `ARIMA`, causal discovery for temporal data

**Installation**: Likely PyCharm environment with standard ML stack. No setup.py provided—projects import modules directly.

## Key Conventions and Patterns

### Data Representation

**Variable Naming**:
- DAG nodes: `X0`, `X1`, `X2`, ... (from DataFrames)
- Internal features: `var_Z`, `var_X`, `var_Y` (non-standard naming in exploratory scripts)
- Features in tensors: Keep dimension alignment with `(num_samples, num_features)`

**Synthetic Data Generation**:
- Noise: `N(0, 1)` standard normal (see `causal_dataset_generator.py`)
- Weights: Uniform `[0.5, 2.0]` for realistic causal effects
- Bias term: Always included in structural equations (`+ ε_i`)

### Model Architecture

**Neural Networks**:
- Simple feedforward MLPs: `Linear → ReLU → Linear`
- Typical sizes: 32-128 hidden units
- Loss: MSE for regression, CrossEntropyLoss for classification
- Optimizer: Adam (lr=0.01 typical)

**Parameter Collection**:
- Extract weights/biases from `model.named_parameters()`
- Store as numpy arrays indexed by `(epoch, layer, param_idx)`
- Use `torch.autograd.functional.jacobian()` for local derivative calculations

### Cross-Component Patterns

**DAG → Dataset → Model Pipeline**:
```python
# 1. Define causal structure
adj_matrix = [[0,1,0], [0,0,1], [0,0,0]]
gen = CausalDatasetGenerator(adj_matrix)

# 2. Generate data respecting causal order
df, equations = gen.generate_dataset_from_dag(1000)

# 3. Train model and collect parameters
X, y = torch.tensor(df[['X0','X1']].values), torch.tensor(df['X2'].values)
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
# ... training with parameter tracking ...
```

**Intervention Framework**:
- **Before**: Train on observational data, collect baseline parameters
- **After**: Apply `do(X_i=0)` (set feature to zero), retrain on intervened data
- **Analysis**: Compare parameter distributions to infer causal mechanisms

## Integration Points

### External Tools Referenced (Not Always Imported)

- **LiNGAM**: Linear Non-Gaussian Acyclic Model (in `LinGam_for_time_series.ipynb`)
- **DoWhy**: Causal inference framework (mentioned in README)
- **Tigramite**: Time series causal discovery (referenced in documentation)
- **CausalEGM**: Encoding generative modeling for causal inference

### Notebook-to-Code Patterns

Exploratory notebooks often contain:
- Parameter sampling logic (becomes `parameter_sample_set()` function)
- Visualization of distributions (histograms, scatter plots)
- Loss tracking during training (collected for convergence analysis)

Extract these into `sandbox_module.py` for reuse across notebooks.

## Critical Files for Understanding Specific Topics

| Topic | Key Files |
|-------|-----------|
| DAG & structural equations | `code_sandbox/causal_dataset_generator.py` |
| Posterior parameter distributions | `code_posterior/posterior_sample_seq.py` |
| Confounding & do-calculus | `code_sandbox/causal_bayes_torch.py`, `causal_bayes_sk-learn.py` |
| Time series methods | `code_sandbox/LinGam_for_time_series.ipynb` |
| Intervention experiments | `code_posterior/posterior_sample_seq.py` (intervention block) |
| Utility functions | `code_sandbox/sandbox_module.py` |

## Common Pitfalls and Solutions

1. **Topological Order**: Always use `predecessors_traversal_order()` to ensure causal order; violating this breaks SCM assumptions
2. **Parameter Shapes**: 3D parameter tensors need `[epoch, output_neurons, input_weights]` indexing
3. **Confounding**: Observational models are biased—always adjust for all confounders or use causal discovery methods
4. **Nonlinearity**: Toggle `nonlinear=True/False` consistently in generator and equations; mismatched causes data-model misalignment

## When Modifying Code

- **Adding new causal discovery method?** Create a new notebook in `code_sandbox/` first, then extract core logic to `sandbox_module.py`
- **Extending parameter estimation?** Follow the 3-step pattern: sample → collect → density-estimate
- **Testing DAG functionality?** Use `pytest` fixtures with simple known-structure DAGs (see `test_causal_dataset_generator.py`)
- **Creating reproducible results?** Set random seeds at notebook/script start: `np.random.seed(0)`, `torch.manual_seed(0)`

