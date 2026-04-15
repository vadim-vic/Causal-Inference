# Numerical experiments on Causal Inference Applications

## Py-files
1. Causal Bayesian Inference from sk-learn [py](causal_bayes_sk-learn.py)
2. Causal Inference with pyTorch [py](causal_bayes_torch.py)
3. Test Module Sandbox [py](sandbox_module.py)

## Generator
1. Generator of synthetic data for causal inference applications [py](generator_causal_dataset.py)
2. u-Tests of the generator [py](test_causal_dataset_generator.py) 
3. Demo of the generator [ipynb](demo_causal_dataset_generator.ipynb)
###
1. Useful module [py](sandbox_module.py) 
2. Notebook for the module [ipynb](sandbox_notebook.ipynb)

## Notebooks  
Below is a list of notebooks and modules for numerical experiments on causal inference applications:
1. Casual Bayesian Inference from sk-learn [ipynb](causal_bayes_sk-learn.ipynb)
2. Causal Inference with DoWhy [ipynb](causal_bayes_dowhy.ipynb)
3. Linear Non-Gaussian Acyclic Model (LiNGAM) for time series data [ipynb](LinGam_for_time_series.ipynb)
4. Causal regression feature selection [ipynb](causal_regression_feature_selection.ipynb)
5. ODE model of Lorentz system [ipynb](model_Lorentz_ODE.ipynb)
6. PDE model of diffusion [ipynb](model_PDE_diffusion.ipynb)
7. Volterra-Lotka model of population dynamics [ipynb](model_Volterra-Lotka model._SCM_DAG.ipynb)
8. Wave equation model [ipynb](model_Wave_equation_SCM.ipynb)
9. ARIMA model for airline passengers [ipynb](sandbox_arima_airline.ipynb)
10. Linear Non-Gaussian Acyclic Model (LiNGAM) for time series data [ipynb](sandbox_lingam.ipynb)
11. Causal inference sandbox [ipynb](sandbox_notebook.ipynb)
12. KNN-based SCM [ipynb](sandbox_scm_knn.ipynb)
13. KNN-based SCM v2 [ipynb](sandbox_scm_knn_v2.ipynb)

# Prize-collecting Steiner Tree: Algorithms for Reconstructing Superposition Trees Based on PCSTFast
The experiments presented in the thesis are contained in [opt_pcst.ipynb](opt_pcst.ipynb).
The file [opt_symb_repr.ipynb](opt_symb_repr.ipynb) also includes functions for generating valid superpositions and code for predicting them.
See also [opt_feature_generators.py](opt_feature_generators.py) and [opt_time_series.py](opt_time_series.py).
The PCSTFast algorithm is used to reconstruct superposition matrices, as described in
1. [A Fast, Adaptive Variant of the Goemans-Williamson Scheme for the Prize-Collecting Steiner Tree Problem](http://people.csail.mit.edu/ludwigs/papers/dimacs14_fastpcst.pdf) by Chinmay Hegde, Piotr Indyk, Ludwig Schmidt // Workshop of the 11th DIMACS Implementation Challenge: Steiner Tree Problems, 2014
2. [A Nearly-Linear Time Framework for Graph-Structured Sparsity](http://people.csail.mit.edu/ludwigs/papers/icml15_graphsparsity.pdf) Chinmay Hegde, Piotr Indyk, Ludwig Schmidt  ICML, 2015
The code opt_psct.ipynb is based on the [PCSTFast](https://github.com/fraenkel-lab/pcst_fast)
The most part of the heuristics in based of solution of thePCST (Prize-Collecting Steiner Tree) are based on the 
> [A General Approximation Technique For Constrained Forest Problems](https://math.mit.edu/~goemans/PAPERS/GoemansWilliamson-1995-AGeneralApproximationTechniqueForConstrainedForestProblems.pdf)  Michel X. Goemans and David E. Williamson SIAM 1995



## See also
1. For parameter estimation sandbox see [the folder](../code_posterior)

