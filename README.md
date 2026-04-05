# Causal inference for time series and text data

## List of Contents
### Texts 
- Likelihood on the CP-DAG in causal discovery [pdf](/slides/text_dag_likelihood.pdf)
- (Draft of the paper) Causal discovery with parameter analysis [pdf](/slides/text_causal_paper.pdf)
- Direct and indirect causes in the time series [pdf](/slides/text_causal_graph.pdf)
- Causal inference in latent space [pdf](/slides/text_causal_latent.pdf) 

### Code
1. MCMC and likelihood of causal graphs [notebook](../code_posterior/causal_graph_likelihood.ipynb)
2. Parameter estimation sandbox [the folder](../code_posterior)
2. Causal inference sandbox [the folder](../code_sandbox)

### Slides
- Causal graphs and optimization [pdf](/slides/intro_CovW.pdf)
- Causal inference in latent space [pdf](/slides/text_causal_latent.pdf)
- Bayesian structure of a model as a causal graph [pdf](/slides/intro_Gamma.pdf)
- Causal inference in latent space [ipynb](/code_posterior/norm_descent_mahalanobis.ipynb)
- Distribution of model parameters and causality (including cones and symbolic) [pdf](/slides/intro_pZW.pdf)
- Estimation of posterior distributions for Bayesian causal inference [in code_posterior](/code_posterior/README.md)
- Discussion of the $p(\mathbf{w}|\text{do}(\mathbf{x}),\mathbf{y})$ for $\mathbf{W}^\mathsf{T}\mathbf{x}$ estimation [pdf](/slides/intro_Setup.pdf)
- Transformers and canonical correlation analysis
for causal inference [pdf](slides/intro_Distr.pdf)
- Canonical correlation analysis for causal inference via generative modeling [pdf](slides/intro_CCA_Gen.pdf)
- Causal inference in the latent space [pdf](slides/intro_CI_CCM.pdf)
- Convergent cross mapping short introduction [pdf](slides/intro_CCM.pdf)
- Causal inference for domain localization and dimensionality reduction [pdf](slides/draft_Causal_Inference.pdf) 
- Code examples for causal inference for 1) [sk-learn](code_sandbox/causal_bayes_sk-learn.py) 2) [pytorch](code_sandbox/causal_bayes_torch.py)

## Collection on causal inference for symbolic regression


## General references
### Team publications
1. [Causal inference for time series](https://causalinferencelab.com/wp-content/uploads/2023/06/Runge_Causal_Inference_for_Time_Series_NREE.pdf), [slides](https://project.inria.fr/aaltd22/files/2022/09/slides_Devijver_AALTD22.pdf)
2. [Causal discovery from time series with hybrids of constraint-based and noise-based algorithms](https://arxiv.org/abs/2306.08765), [appendix](https://arxiv.org/abs/2306.08765)
3. [Identifiability in Causal Abstractions: A Hierarchy of Criteria](https://arxiv.org/pdf/2507.06213)
4. [On the Fly Detection of Root Causes from Observed Data with Application to IT Systems](https://arxiv.org/abs/2402.06500)
5. [Relaxing partition admissibility in Cluster-DAGs: a causal calculus with arbitrary variable clustering](https://arxiv.org/abs/2511.01396)
6. [Modèles de Fondation et Ajustement : Vers une Nouvelle Génération de Modèles pour la Prévision des Séries Temporelles](https://www.arxiv.org/abs/2511.22674)
7. [Causal Discovery from Time Series with Hybrids of Constraint-Based and Noise-Based Algorithms](https://arxiv.org/abs/2306.08765)
8. [Inferring extended summary causal graphs from observational time series](https://arxiv.org/pdf/2205.09422)
9.  [Complete Characterization for Adjustment in Summary Causal Graphs of Time Series](https://proceedings.mlr.press/v286/yvernes25a.html), 2025
11. [Identifiability by common backdoor in summary causal graphs of time series](https://arxiv.org/abs/2506.14862)
12. [Discovery of extended summary graphs in time series](https://proceedings.mlr.press/v180/assaad22a.html), 2022 (discussed)
13. [Causal Representation Learning from Multiple Distributions: A General Setting](https://arxiv.org/pdf/2402.05052), 2024 (discussed-twice!)
10. [Learning Causal Graphs with Small Interventions](https://arxiv.org/abs/1511.00041), 2015 (discussed)
<!-- https://arxiv.org/abs/2204.06479 (no need to cite) -->

### Textbooks on causal inference
1. [Causal Artificial Intelligence](https://causalai-book.net/) by Elias Bareinboim
1. [Elements of Causal Inference: Foundations and Learning Algorithms](https://web.math.ku.dk/~peters/jonas_files/ElementsOfCausalInference.pdf) by Jonas Peters, Dominik Janzing, and Bernhard Schölkopf
2. [The book of why: the new science of cause and effect](https://bayes.cs.ucla.edu/WHY/) by Judea Pearl and Dana Mackenzie
3. [Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/landing-page.html) site by Matheus Facure Alves
4. [The Effect: An Introduction to Research Design and Causality](https://theeffectbook.net/) site by Nick Huntington-Klein
5. [Introduction to Causal Inference](https://www.bradyneal.com/causal-inference-course) course by Brady Neal and [book](https://www.bradyneal.com/Introduction_to_Causal_Inference-Dec17_2020-Neal.pdf)
6. [Causal Inference in Statistics: A Primer](https://web.cs.ucla.edu/~kaoru/primer-complete-2019.pdf) Judea Pearl, Madelyn Glymour, and Nicholas P. Jewell
7. [Introduction to Causal Inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC2836213) by Judea Pearl
8. [Probabilistic Graphical Models: Principles and Techniques](http://mcb111.org/w06/KollerFriedman.pdf) by Daphne Koller and Nir Friedman, this fat gray book
9. [A Tutorial on Learning With Bayesian Networks](http://heckerman.com/david/tutorial.pdf) by David Heckerman
10. [Introduction to Judea Pearl’s Do-Calculus](https://arxiv.org/abs/1305.5506) by Robert R. Tucci
11. inFERENCe by Ferenc Huszár: [part 1](https://www.inference.vc/untitled/), [part 4](https://www.inference.vc/causal-inference-4/), [example](https://www.inference.vc/causal-inference-2-illustrating-interventions-in-a-toy-example/) 

### General references on time series causal inference
1. [Toward Causal Representation Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9363924) by Bernhard Schölkopf et al.
2. [A Linear Non-Gaussian Acyclic Model for Causal Discovery](https://www.cs.helsinki.fi/group/neuroinf/lingam/JMLR06.pdf)
2. [Causal Structure Learning and Effect Identification in Linear Non-Gaussian Models and Beyond](https://helda.helsinki.fi/server/api/core/bitstreams/db658dd0-6d6e-4e35-854f-f4d973548599/content) by Doris Entner, and [site](https://sites.google.com/site/dorisentner/publications?authuser=0#h.p_pKxBzrRlMHl_)
3. [Causal inference for time series](https://causalinferencelab.com/wp-content/uploads/2023/06/Runge_Causal_Inference_for_Time_Series_NREE.pdf)
5. [LiNGAM: References](https://sites.google.com/view/sshimizu06/lingam)
6. [Causal inference for time series](https://causalinferencelab.com/publications/) by Jakob Runge et al., and [site](https://causalinferencelab.com/)

### Tools for causal discovery
1. [LiNGAM: Linear Non-Gaussian Acyclic Model for Causal Discovery](https://lingam.readthedocs.io/), [github](https://github.com/cdt15/lingam)
2. [Do-Why: is a Do-sampler](https://www.pywhy.org/dowhy/v0.11.1/example_notebooks/do_sampler_demo.html), [site](https://www.pywhy.org/), [pypi](https://pypi.org/project/dowhy/), [pywhy](https://www.pywhy.org/dowhy/v0.14/), [do-sampler](https://www.pywhy.org/dowhy/v0.11.1/example_notebooks/do_sampler_demo.html)
3. [Tigramite and PCMCI](https://jakobrunge.github.io/tigramite/)
4. [F-PCMCI](https://lcastri.github.io/fpcmci/)
5. [causal-learn](https://causal-learn.readthedocs.io/)
6. [CausalEGM](https://causalegm.readthedocs.io/en/latest/installation.html)
7. [Tetrad](https://tetrad-manual.readthedocs.io/en/latest/workflows/workflows.html)
#### Code
1. [Time series auto-former enc dec](https://github.com/rakuyorain/Implicit-Forecaster/blob/main/layers/Autoformer_EncDec.py)
2. [Py-tetrad](https://github.com/cmu-phil/py-tetrad/tree/main/pytetrad/resources)]

## Special references 

### Generative modeling for causal inference
1. [Causal EGM: encoding generative modeling](https://causalegm.readthedocs.io/en/latest/tutorial_r.html)
2. Generative CCA: [Deep Dynamic Probabilistic Canonical Correlation Analysis](https://arxiv.org/html/2502.05155v1)
3. [Flow-Based Non-stationary Temporal Regime Causal Structure Learning](https://arxiv.org/abs/2506.17065), 2025

### Non-Parametric CI
1. [Non-Parametric Path Analysis in Structural Causal Models](https://par.nsf.gov/servlets/purl/10111046)
2. [Non-Parametric Methods for Partial Identification of Causal Effects](https://causalai.net/r72.pdf)
3. [Non-parametric Causal Inference in Dynamic Thresholding Designs](https://arxiv.org/abs/2512.15244)

### Kernel-based CI
1. [Practical Kernel Selection for Kernel-based Conditional Independence Test](https://neurips.cc/virtual/2025/loc/san-diego/poster/118068})

### Functional CI with brain
1. [Functional CI with brain](https://arxiv.org/pdf/2401.09641)
2. [Modeling Causal Interactions Across Brain Functional Subnetworks](https://arxiv.org/pdf/2511.05548)

### Structure learning
1. [DAGs with NO TEARS: Continuous Optimization for Structure Learning](https://proceedings.neurips.cc/paper_files/paper/2018/file/e347c51419ffb23ca3fd5050202f9c3d-Paper.pdf)

### Granger 
1. [Learning Causal Representations with Granger PCA](https://openreview.net/pdf?id=XsTEnaD_Lel)

### Convergent Cross Mapping and Sugihara
1. [Causal Discovery in Symmetric Dynamic Systems with Convergent Cross Mapping](https://arxiv.org/html/2505.04815v1)
2. [Convergent Cross-Mapping and Pairwise Asymmetric Inference](https://arxiv.org/pdf/1407.5696)
3. [Limits to Causal Inference with State-Space Reconstruction for Infectious Disease](https://pmc.ncbi.nlm.nih.gov/articles/PMC5193453/)
4. [Distinguishing time-delayed causal interactions using convergent cross mapping](https://www.nature.com/articles/srep14750)
5. [Causal Modelling and Brain Connectivity in Functional Magnetic Resonance Imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC2642881/)
6. Science 2012, Foundational CCM method: Detecting causality in complex ecosystems
7. Science 2016, Advanced CCM/forecasting techniques: Information leverage in interconnected ecosystems
8. Nature Scientific Reports 2015, Distinguishing time-delayed causal interactions using CCM
9. [Improved convergent cross mapping method for causal inference based on decomposition of the Lorenz trajectory](https://www.nature.com/articles/s41598-025-22300-y)
9. [Causal Discovery in Semi-Stationary Time Series](https://arxiv.org/pdf/2407.07291)
10. [Towards Characterizing Domain Counterfactuals for Invertible Latent Causal Models](https://openreview.net/forum?id=v1VvCWJAL8)
11. [Conditional Generative Models are Sufficient to Sample from Any Causal Effect Estimand](https://arxiv.org/pdf/2402.07419)
12. [Causal Bandits](https://github.com/CausalML-Lab/CausalBandits_with_UnknownGraph)
13. [Detecting causal associations in large nonlinear time series datasets](https://arxiv.org/pdf/1702.07007)
14. [Causalized Convergent Cross Mapping and Its Implementation in Causality Analysis](https://www.mdpi.com/1099-4300/26/7/539/pdf?version=1719219192)
15. [Causalized convergent cross-mapping and its approximate equivalence with directed information in causality analysis](https://academic.oup.com/pnasnexus/article-pdf/3/1/pgad422/55381360/pgad422.pdf)
16. [Latent Convergent Cross Mapping](https://openreview.net/forum?id=4TSiOTkKe5P), 2021 

### Runge PCMCI and time series
1. [Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets](https://proceedings.mlr.press/v124/runge20a/runge20a.pdf)
2. Science advances, 2019: [Detecting and quantifying causal associations in large nonlinear time series datasets],[pdf 2019](https://www.science.org/doi/epdf/10.1126/sciadv.aau4996), [arxiv 2017](https://arxiv.org/abs/1702.07007), [arxiv 2018](https://arxiv.org/pdf/1702.07007)
3. An Algorithm for Fast Recovery of  Sparse Causal Graphs by Peter Spirtes and Clark Glymour, 1990 [original PC](https://www.cmu.edu/dietrich/philosophy/docs/tech-reports/15_Spirtes.pdf)

### Generative modeling for CI
1. Causal Effect Inference with Deep Latent-Variable Models [arxiv](https://arxiv.org/abs/1705.08821)
2. An encoding generative modeling approach to dimension  reduction and covariate adjustment in causal inference  with observational studies [doi](https://pnas.org/doi/10.1073/pnas.2322376121)
2. GANITE [pdf](https://pmc.ncbi.nlm.nih.gov/articles/PMC7759680/pdf/fgene-11-585804.pdf), [pdf](https://openreview.net/pdf?id=ByKWUeWA-)
3. Conditional GANs [pdf](https://pmc.ncbi.nlm.nih.gov/articles/PMC7759680/pdf/fgene-11-585804.pdf), 2020
4. A Conditional Mutual Information Estimator [pdf](https://www.mdpi.com/1099-4300/24/9/1234), 2022

### Nature
1. Data-driven control of complex networks [doi](https://doi.org/10.1038/s41467-021-21554-0), 2021
2.  Causal chambers as a real-world physical testbed for AI methodology [doi](https://doi.org/10.1038/s42256-024-00964-x), 2025
3. Improved convergent cross  mapping method for causal  inference based on decomposition [doi](https://www.nature.com/articles/s41598-025-22300-y), 2025

### Functional Data Analysis and CI
1. [Fast fitting of neural ordinary differential equations by  Bayesian neural gradient matching](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.14121), 2022

### Basics of causal discovery
1. [Do-Calculus: Causal Inference Rules](https://www.emergentmind.com/topics/do-calculus), 2025
2. [do(x) operator meaning](https://stats.stackexchange.com/questions/211008/dox-operator-meaning)
3. [Do-calculus rules and backdoors](https://www.andrewheiss.com/blog/2021/09/07/do-calculus-backdoors/)
4. [Causal Models](https://plato.stanford.edu/entries/causal-models/index.html) in the Stanford Encyclopedia of Philosophy, 2018
2. [Back and frontdoor](https://stats.stackexchange.com/questions/312992/causal-effect-by-back-door-and-front-door-adjustments)
2. [The Do-Calculus Revisited](https://ftp.cs.ucla.edu/pub/stat_ser/r402.pdf) by Judea Pearl, 2012
3. [ML beyond Curve Fitting: An Intro to Causal Inference and do-Calculus](https://www.inference.vc/untitled/) by Ferenc Huszár, 2018
3. [A Survey on Causal Discovery: Theory and Practice](https://arxiv.org/pdf/2305.10032), 2025

### Natural gradient descent and Geometric Learning [CMA-ES](https://en.wikipedia.org/wiki/CMA-ES)
1. [Exponential Natural Evolution Strategies](https://people.idsia.ch/~tom/publications/xnes.pdf), 2010
2. [Information-Geometric Optimization Algorithms](https://www.jmlr.org/papers/volume18/14-467/14-467.pdf), 2017
3. [Natural Gradients](https://andrewcharlesjones.github.io/journal/natural-gradients.html) by Andrew Charles Jones, 2018
4. (Bayesian optimization)[https://github.com/bayesian-optimization/BayesianOptimization]

### Dynamic systems and CI
1. [From Deterministic ODEs to Dynamic Structural Causal Models](https://arxiv.org/pdf/1608.08028)
2. [Causal Modeling of Dynamical Systems](https://arxiv.org/pdf/1803.08784)
3. [Implicit Reasoning in Deep Time Series Forecasting](https://arxiv.org/abs/2409.10840)
4. [Towards Accurate Time Series Forecasting via Implicit Decoding](https://neurips.cc/virtual/2025/loc/san-diego/poster/116698)








<!---### LLM and CI (obsoleted) 
Integrating Large Language Models in Causal Discovery: A Statistical Causal Approach
https://arxiv.org/html/2402.01454v4
-->


