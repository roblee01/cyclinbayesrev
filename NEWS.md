# cyclinbayes 1.0.0

## Initial Release

### Bayesian Causal Discovery
- Implemented BayesDAG for Bayesian causal discovery in directed acyclic graphs.
- Implemented BayesDCG for Bayesian causal discovery in directed cyclic graphs.
- Supports linear structural equation models with non-Gaussian errors.
- Provides posterior samples of graph structures and causal effect coefficients.

### Posterior Inference
- Added point_est_graph() for decision-theoretic posterior graph selection.
- Added posterior_interval_est() for credible and HPD interval estimation.
- Added posterior_network_motif() for posterior network motif analysis.

### Software Implementation
- Implemented computational routines using Rcpp and RcppArmadillo.
- Added documentation and reproducible examples.
- Added automated tests for principal package functions.
- Established continuous integration using GitHub Actions.
- Verified R CMD checks across macOS, Windows, and Linux.
