# Estimation of Posterior Distributions for Bayesian Causal Inference

The posterior distribution $p(\mathbf{w}|\text{do}(\mathbf{x}),\mathbf{y})$ is a function over the support set $\{\mathbf{w}\}$. 
It is estimated using sampling technique:
1. Sample the dataset $\{\mathbf{x}_i,\mathbf{y}_i\}$. Use the index $i$ for unform sampling $i\in\{1,\dots N\}$. Make set of batches.
2. For each batch, estimate the parameters $\mathbf{w}_i$ to maximize 1) likelihood, 2) posterior.
3. Collect the set of parameters $\{\mathbf{w}_i\}$ over all batches.
4. Estimate the density of the set $\{\mathbf{w}_i\}$ using 1) gaussian model, 2) flow-based model.

List of notebooks and modules 
1. Sampling of parameters from prior distribution of a neural network [ipynb](posterior_sample_1nn.ipynb)
2. Optimizing to sample of parameters from prior distribution of a 2nn [ipynb](posterior_sample_2nn.ipynb)
3. Estimation of posterior distribution using flow-based model [ipynb](posterior_flow_density.ipynb)
4. Visualisation of posterior distribution in Riemannian space [ipynb](posterior_visualization.ipynb)
5. Singular Values Decomposition as part of neural network [ipynb](decomposition_ssa_nn.ipynb)