
<!-- README.md is generated from README.Rmd. Please edit that file -->

# cyclinbayes

<!-- badges: start -->

[![R-CMD-check](https://github.com/roblee01/cyclinbayesrev/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/roblee01/cyclinbayesrev/actions/workflows/R-CMD-check.yaml)

[![License:
MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/roblee01/cyclinbayes/blob/main/LICENSE.md)

[![GitHub
issues](https://img.shields.io/github/issues/roblee01/cyclinbayes)](https://github.com/roblee01/cyclinbayes/issues)

<!-- badges: end -->

`cyclinbayes` is an open-source R package for Bayesian causal discovery
in directed acyclic graphs (DAGs) and directed cyclic graphs (DCGs)
under linear non-Gaussian structural equation models. The package
provides posterior inference for graph structures and causal effects,
allowing users to quantify uncertainty in causal relationships,
including those involving feedback cycles.

The package implements two MCMC-based algorithms: `BayesDAG()` for
acyclic structures using collapsed Gibbs sampling and `BayesDCG()` for
cyclic structures using joint Metropolis-Hastings updates. Both methods
use hierarchical priors to encourage sparse graphs.

In addition to graph estimation, `cyclinbayes` provides posterior edge
inclusion probabilities, credible intervals for causal effects, network
motif probabilities, and decision-theoretic graph selection. The package
is implemented using Rcpp and RcppArmadillo for computational
efficiency.

## Installation from Github

You can install the development version of `cyclinbayes` from GitHub
using remotes:

``` r
#install.packages("remotes")
#remotes::install_github("roblee01/cyclinbayesrev")
```

Then load the package:

``` r
library(cyclinbayesrev)
library(ggplot2)
#> Warning: package 'ggplot2' was built under R version 4.5.2
library(igraph)
#> Warning: package 'igraph' was built under R version 4.5.2
#> 
#> Attaching package: 'igraph'
#> The following objects are masked from 'package:stats':
#> 
#>     decompose, spectrum
#> The following object is masked from 'package:base':
#> 
#>     union

# If using dist_type = "sid" and you see missing RBGL/graph:
#install.packages("BiocManager")
#BiocManager::install(c("graph", "RBGL"))
```

## Acyclic (DAG) Example

Below is a simple example demonstrating how to use `BayesDAG()`, the
Bayesian LiNGAM sampler for directed acyclic graph (DAG) structure
estimation. Let $p$ denote the number of variables, $N$ the sample size,
and `num_iter` the number of MCMC iterations. Let $i \in \{1,\ldots,p\}$
index variables and $q \in \{1,\ldots,N\}$ index observations.

We generate structural errors from a finite Gaussian mixture model,

$$\epsilon_i^{(q)}
\sim
\sum_{k=1}^{M}
\pi_{ik}\,N(\mu_{ik},\tau_{ik}),$$

with

$$M=2,\qquad
(\mu_{i1},\mu_{i2})=(-0.5,0.5),\qquad
(\tau_{i1},\tau_{i2})=(0.1,0.3),\qquad
(\pi_{i1},\pi_{i2})=(0.5,0.5).$$

We generate a sparse DAG by including each candidate directed edge with
probability $1-\Delta=0.1$, where $\Delta=0.9$, while enforcing
acyclicity. For each included edge $j\to i$, the corresponding nonzero
causal effect coefficient is generated according to

$$|B_{ij}| \sim \mathrm{Unif}(0.4,0.9),$$

with its sign chosen independently to be positive or negative with equal
probability. For excluded edges, $B_{ij}=0$.

Given the causal effect matrix $B$ and structural error matrix
$\epsilon$, the observed data are generated from

$$Y=(I-B)^{-1}\epsilon,$$

where the $i$-th row of $Y$ corresponds to

$$(Y_i^{(1)},\ldots,Y_i^{(N)})^\top.$$

``` r
#######################################
# Simulation and MCMC settings
#######################################
N = 200                     # Sample size of the simulated example dataset
num_covariates = 10         # Number of variables in the simulated example dataset
M = 5                       # Number of components in the fitted finite Gaussian mixture model
num_iter = 100000           # Total number of MCMC iterations
burn_in_iterations = 20000  # Number of initial MCMC iterations discarded as burn-in

#######################################
# Hyperparameter setup
#######################################
params = list(
  a_mu      = 0,
  b_mu      = 2,
  a_gamma   = 0.5,
  b_gamma   = 0.5,
  a_gamma_1 = 2,
  b_gamma_1 = 1,
  a_og_tao = 0.01,
  b_og_tao = 0.01,
  a_tao     = 2,
  b_tao     = 1,
  alpha     = 1
)

#######################################
# Generate DAG Example
#######################################
example_list = generates_examples_DAG(
  num_covariates = num_covariates,
  N              = N,
  M_input        = 2,             # Number of Gaussian mixture components used to generate errors
  prob_sparsity  = 0.90,          # Probability of no edge (edge probability = 0.10)
  seed_input     = 21,            # Random seed for data generation
  mag_range      = c(0.4, 0.9),   # Range of absolute values of nonzero edge weights
  prob_positive  = 0.5,           # Probability that a nonzero edge weight is positive
  seed_structure = 1,             # Random seed for generating the DAG structure
  seed_weights   = 20             # Random seed for generating edge weights
)

data_matrix = example_list$data_matrix
Adjacency_matrix_true = example_list$Adjacency_matrix_true
Causal_effect_matrix_true = example_list$Causal_effect_matrix_true
Z_matrix_true = example_list$Z_matrix_true
```

Before examining posterior summaries, it is helpful to visualize the
true underlying DAG used in the simulation. This provides a direct point
of comparison for the estimated graph structures returned by the
sampler. The plot below displays the ground truth adjacency structure,
where each directed edge represents a causal relationship from one
variable to another.

``` r
#######################################
# Build directed graph from adjacency matrix
#######################################

g_true = igraph::graph_from_adjacency_matrix(
Adjacency_matrix_true,
mode = "directed",
diag = FALSE
)

plot(
  g_true,
  vertex.size = 20,
  vertex.label.cex = 0.8,
  edge.arrow.size = 0.5,
  main = "True DAG Structure"
)
```

<img src="man/figures/README-unnamed-chunk-5-1.png" alt="" width="100%" />

The sampler runs for `num_iter` iterations, but only post burn-in draws
are returned for posterior analysis. The `burn_in_iterations` argument
specifies how many initial iterations are discarded. Because the DAG
sampler uses an annealing contribution during the early iterations, the
implementation ensures that the effective burn-in is at least 20% of
`num_iter`. If a smaller value is supplied, it is automatically
increased to the end of this annealing window. Thus, the returned
samples correspond only to the post-annealing posterior sampling phase.

Let $t=1,\ldots,m$ index the $m$ retained posterior draws. For each
retained draw $t$, the output includes:

- Adjacency matrix $E^{(t)}\in\{0,1\}^{p\times p}$
- Causal effect matrix $B^{(t)}\in\mathbb{R}^{p\times p}$
- Mixture parameters for the error model, including the
  component-specific means and variances
  $\mu^{(t)}=(\mu_{ik}^{(t)})\in\mathbb{R}^{p\times M}$ and
  $\tau^{(t)}=(\tau_{ik}^{(t)})\in\mathbb{R}^{p\times M}$, where
  $i=1,\ldots,p$ indexes variables and $k=1,\ldots,M$ indexes mixture
  components.

For posterior summaries, the retained matrices are flattened
(vectorized) into parameter vectors. Rows correspond to retained
posterior draws and columns correspond to fixed matrix entries, for
example one column for $B_{ij}$ or $\mu_{ik}$. This makes edge
probabilities, HPD/credible intervals, and other posterior summaries
straightforward to compute.

``` r
#######################################
# Parameter Initialization
#######################################
B_seed = directlingam_seed(data_matrix)  # Initial coefficient matrix estimated using DirectLiNGAM
seed_edge_threshold = 0.05  # Threshold for removing small absolute edge weights
seed_rho_target = 0.95      # Maximum target spectral radius for rescaling the coefficient matrix
seed_settle_sweeps = 30     # Number of Gibbs sampling sweeps for mixture initialization
#######################################
# Initialization
#######################################  
init_state = init_from_seed(
  B_seed, data_matrix, N, num_covariates, M,
  params$a_mu, params$b_mu, params$a_tao, params$b_tao, params$alpha,
  params$a_gamma, params$b_gamma, params$a_gamma_1, params$b_gamma_1,
  edge_threshold = seed_edge_threshold,
  rho_target     = seed_rho_target,
  settle_sweeps  = seed_settle_sweeps
)
#> init_from_seed: edges 9  rho 0.000  gamma_1 0.526  gamma_result 0.104  log-post 46.7
#######################################
# Run Bayesian DAG Sampler
#######################################

results_list = BayesDAG(
      data_matrix,
      params$a_mu, params$b_mu,
      params$a_gamma, params$b_gamma,
      params$a_tao, params$b_tao,
      params$a_og_tao, params$b_og_tao,
      params$a_gamma_1, params$b_gamma_1,
      params$alpha,
      M,
      num_iter,
      burn_in_iterations = burn_in_iterations,
      init_Adjacency     = init_state$Adjacency_matrix,
      init_Causal_effect = init_state$Causal_effect_matrix,
      init_mu            = init_state$mu_mat,
      init_tao           = init_state$tao_mat,
      init_pi            = init_state$pi_mat,
      init_Z             = init_state$Z_matrix,
      init_gamma_1       = init_state$gamma_1,
      init_gamma_result  = init_state$gamma_result
    )
#> [BayesDAG] 25%  (25000/100000)
#> [BayesDAG] 50%  (50000/100000)
#> [BayesDAG] 75%  (75000/100000)
#> [BayesDAG] done -- 100000 iterations

#######################################
# Extract posterior outputs
#######################################
log_likelihood_list = results_list$log_likelihood_list
Adjacency_matrix_list = results_list$Adjacency_matrix_list 
Causal_effect_matrix_list = results_list$Causal_effect_matrix_list
gamma_list = results_list$gamma_list
gamma_1_list = results_list$gamma_1_list
mu_matrix_list = results_list$mu_matrix_list
tao_matrix_list = results_list$tao_matrix_list
pi_matrix_list = results_list$pi_matrix_list

# Information about the retained posterior sample
first_stored_iteration = results_list$first_stored_iteration
posterior_sample_length = results_list$n_stored

first_stored_iteration
#> [1] 20001
posterior_sample_length
#> [1] 80000
```

All downstream posterior summaries use these retained post burn-in
draws. In this example, `burn_in_iterations = 20000`, so the returned
posterior samples begin at iteration 20,001 and contain 80,000 draws.

To obtain a representative estimate of the graph structure, we use the
function `point_est_graph(),` which selects the posterior weighted
medoid, the graph that minimizes the weight distance to all other
sampled adjacency matrices. Users may select one of the built in
distances or supply their own custom functions. The available distance
metrics are:

- **Structural Hamming Distance (SHD):**  
  Counts the number of single edge edits (additions, deletions, or
  reversals) required to transform one graph into another.

- **Structural Intervention Distance (SID):**  
  Measures how many node pairs $(i, j)$ imply different intervention
  distributions P($\mathbf{Y}_j$ \| do($\mathbf{Y}_i$)).  
  (Applicable only when the posterior graphs are DAGs)

Users may also specify

``` r
dist_type = 'custom'
dist_fun = function(A,B){...}
```

where $A$ and $B$ are $p\times p$ adjacency matrices. The function must
return a non negative scalar distance.

``` r
#############################################
# Best Graph Structure determined through shd
#############################################
Adjacency_matrix_shd = point_est_graph(Adjacency_matrix_list, dist_type = 'shd')
Adjacency_matrix_shd
#>       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#>  [1,]    0    0    0    0    0    0    0    0    0     0
#>  [2,]    0    0    0    0    0    0    0    0    1     0
#>  [3,]    0    1    0    1    0    0    0    0    0     0
#>  [4,]    0    0    0    0    0    0    0    0    0     0
#>  [5,]    1    0    0    0    0    0    0    0    0     0
#>  [6,]    0    1    0    0    0    0    0    0    0     0
#>  [7,]    0    0    0    0    0    0    0    0    0     0
#>  [8,]    0    0    0    1    0    0    0    0    0     0
#>  [9,]    0    0    0    0    0    0    0    0    0     0
#> [10,]    0    0    0    1    1    0    1    0    0     0
```

``` r
############################################
# Best Graph Structure determined through sid
############################################
if (requireNamespace("SID", quietly = TRUE)) {
  Adjacency_matrix_sid = point_est_graph(Adjacency_matrix_list, dist_type = "sid")
  Adjacency_matrix_sid
} else {
  message("SID not installed. Install SID (and possibly Bioconductor graph/RBGL) to run SID.")
}
#>       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#>  [1,]    0    0    0    0    0    0    0    0    0     0
#>  [2,]    0    0    0    0    0    0    0    0    1     0
#>  [3,]    0    1    0    1    0    0    0    0    1     0
#>  [4,]    0    0    0    0    0    0    0    0    0     0
#>  [5,]    1    0    0    0    0    0    0    0    1     0
#>  [6,]    0    1    0    0    0    0    0    0    0     0
#>  [7,]    0    0    0    0    0    0    0    0    0     0
#>  [8,]    0    0    0    1    0    0    0    0    0     0
#>  [9,]    0    0    0    0    0    0    0    0    0     0
#> [10,]    1    0    0    1    1    0    1    0    0     0
```

As an example for a custom function, we could count the total number of
edge mismatches.

``` r
custom_edge_mismatch = function(A, B) {
  return(sum(abs(A - B)))
}

Adjacency_matrix_custom = point_est_graph(Adjacency_matrix_list, dist_type = 'custom', dist_fun = custom_edge_mismatch)
Adjacency_matrix_custom
#>       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#>  [1,]    0    0    0    0    0    0    0    0    0     0
#>  [2,]    0    0    0    0    0    0    0    0    1     0
#>  [3,]    0    1    0    1    0    0    0    0    0     0
#>  [4,]    0    0    0    0    0    0    0    0    0     0
#>  [5,]    1    0    0    0    0    0    0    0    0     0
#>  [6,]    0    1    0    0    0    0    0    0    0     0
#>  [7,]    0    0    0    0    0    0    0    0    0     0
#>  [8,]    0    0    0    1    0    0    0    0    0     0
#>  [9,]    0    0    0    0    0    0    0    0    0     0
#> [10,]    0    0    0    1    1    0    1    0    0     0
```

We observe that the structures selected using SHD and the custom edge
mismatch distance are identical, whereas the structure selected using
SID differs.

``` r
Adjacency_matrix_true
#>       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#>  [1,]    0    0    0    0    0    0    0    0    0     0
#>  [2,]    0    0    0    0    0    0    0    0    1     0
#>  [3,]    0    1    0    1    0    0    0    0    0     0
#>  [4,]    0    0    0    0    0    0    0    0    0     0
#>  [5,]    1    0    0    0    0    0    0    0    0     0
#>  [6,]    0    1    0    0    0    0    0    0    0     0
#>  [7,]    0    0    0    0    0    0    0    0    0     0
#>  [8,]    0    0    0    1    0    0    0    0    0     0
#>  [9,]    0    0    0    0    0    0    0    0    0     0
#> [10,]    0    0    0    1    1    0    1    0    0     0
```

In addition to selecting a representative graph, users can evaluate
posterior support for a specified network motif using
`posterior_network_motif()`. The function computes the proportion of
posterior graph samples containing all edges in the specified motif.
Here, we evaluate support for the edges in the true DAG:

``` r
true_graph_structure = igraph::graph_from_adjacency_matrix(Adjacency_matrix_true)
posterior_network_motif(true_graph_structure, Adjacency_matrix_list)
#> [1] 1
```

A value of 1 indicates that all edges in the specified true-graph motif
are present in every retained posterior graph sample. To assess sampler
behavior, we next plot the log likelihood over the retained posterior
sampling phase.

``` r
posterior_iterations = seq(
  from = first_stored_iteration,
  length.out = posterior_sample_length
)

plot(
  posterior_iterations,
  log_likelihood_list,
  type = 'l',
  xlab = 'Iterations',
  ylab = 'log likelihood values'
)
```

<img src="man/figures/README-unnamed-chunk-12-1.png" alt="" width="100%" />

As shown above, the log-likelihood remains within a relatively stable
range during the retained posterior-sampling phase, providing empirical
evidence that the sampler has entered a stable sampling regime. The
graph summaries and interval estimates below are therefore computed
directly from the post burn-in samples returned by `BayesDAG()`.

Posterior interval estimates are obtained using
`posterior_interval_est()`, which computes highest posterior density
(HPD) intervals and equal tailed credible intervals column wise for each
element of the sampled parameter matrices. To illustrate posterior
uncertainty in the causal effect coefficients, we examine the nonzero
entries of the true causal effect matrix $B$.

The figures below display the posterior estimates together with their
95% HPD and equal-tailed credible intervals.For each causal edge, the
red $\color{red}{\times}$ marks the true value of the corresponding
causal effect coefficient, while the black point and vertical interval
summarize the posterior estimate and uncertainty. This allows each
estimated effect to be compared directly with its ground-truth value.
Overall, the posterior estimates lie close to the true effects, and the
intervals generally cover the corresponding ground truth values,
indicating accurate recovery of the causal effect magnitudes in this
example.

``` r
#######################################
# Posterior intervals
#######################################

Causal_effect_matrix_summary =
  posterior_interval_est(Causal_effect_matrix_list, level = 0.95)

hpd_matrix_acyclic = Causal_effect_matrix_summary$hpd_matrix
ci_matrix_acyclic  = Causal_effect_matrix_summary$ci_matrix


#######################################
# True nonzero causal effects
#######################################

# R vectorizes matrices column-by-column
true_vec = as.vector(Causal_effect_matrix_true)

# Indices of all truly nonzero causal effects
true_idx = which(true_vec != 0)

# Convert vector indices back to matrix coordinates
coords = arrayInd(
  true_idx,
  .dim = dim(Causal_effect_matrix_true)
)

# Convention: B_ij is the effect j -> i
true_effects_df = data.frame(
  coef_index = true_idx,
  child  = coords[, 1],
  parent = coords[, 2],
  truth  = true_vec[true_idx]
)

true_effects_df$edge = paste0(
  true_effects_df$parent,
  " -> ",
  true_effects_df$child
)

#true_effects_df


#######################################
# Check dimensions before matching
#######################################

if (ncol(hpd_matrix_acyclic) != length(true_vec)) {
  stop("HPD matrix ordering/dimensions do not match Causal_effect_matrix_true.")
}

if (nrow(ci_matrix_acyclic) != length(true_vec)) {
  stop("CI matrix ordering/dimensions do not match Causal_effect_matrix_true.")
}


#######################################
# HPD intervals for TRUE causal effects
#######################################

hpd_sub = t(
  hpd_matrix_acyclic[, true_idx, drop = FALSE]
)

colnames(hpd_sub) = c("lower", "upper")

data_hpd = data.frame(
  edge  = true_effects_df$edge,
  truth = true_effects_df$truth,
  lower = hpd_sub[, "lower"],
  upper = hpd_sub[, "upper"]
)

# Midpoint of HPD interval
data_hpd$estimate =
  (data_hpd$lower + data_hpd$upper) / 2

# Keep edge order fixed
data_hpd$edge =
  factor(data_hpd$edge, levels = data_hpd$edge)


#######################################
# Plot HPD intervals
#######################################

ggplot(data_hpd, aes(x = edge, y = estimate)) +
  geom_errorbar(
    aes(ymin = lower, ymax = upper),
    width = 0.2
  ) +
  geom_point(size = 3) +

  # True causal effect
  geom_point(
    aes(y = truth),
    shape = 4,
    size = 4,
    stroke = 1.3,
    color = "red"
  ) +

  labs(
    x = "True Causal Edge",
    y = "Causal Effect",
    title = "Posterior Estimates with 95% HPD Intervals"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    )
  )


#######################################
# 95% credible intervals for TRUE effects
#######################################

ci_sub =
  ci_matrix_acyclic[true_idx, , drop = FALSE]

data_ci = data.frame(
  edge     = true_effects_df$edge,
  truth    = true_effects_df$truth,
  lower    = ci_sub[, 1],
  estimate = ci_sub[, 2],
  upper    = ci_sub[, 3]
)

data_ci$edge =
  factor(data_ci$edge, levels = data_ci$edge)


#######################################
# Plot credible intervals
#######################################

ggplot(data_ci, aes(x = edge, y = estimate)) +
  geom_errorbar(
    aes(ymin = lower, ymax = upper),
    width = 0.2
  ) +
  geom_point(size = 3) +

  # True causal effect
  geom_point(
    aes(y = truth),
    shape = 4,
    size = 4,
    stroke = 1.3,
    color = "red"
  ) +

  labs(
    x = "True Causal Edge",
    y = "Causal Effect",
    title = "Posterior Estimates with 95% Credible Intervals"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    )
  )


#######################################
# Coverage
#######################################

data_hpd$covered =
  data_hpd$truth >= data_hpd$lower &
  data_hpd$truth <= data_hpd$upper

data_ci$covered =
  data_ci$truth >= data_ci$lower &
  data_ci$truth <= data_ci$upper

cat(
  "HPD coverage:",
  sum(data_hpd$covered),
  "out of",
  nrow(data_hpd),
  "\n"
)
#> HPD coverage: 9 out of 9

cat(
  "Credible interval coverage:",
  sum(data_ci$covered),
  "out of",
  nrow(data_ci),
  "\n"
)
#> Credible interval coverage: 9 out of 9
```

<img src="man/figures/README-unnamed-chunk-13-1.png" alt="" width="100%" /><img src="man/figures/README-unnamed-chunk-13-2.png" alt="" width="100%" />

In this acyclic example, the posterior estimates closely match the true
causal effect coefficients across all true edges. The 95% HPD and equal
tailed credible intervals contain, or closely surround, the
corresponding true values, indicating accurate recovery of both the
direction and magnitude of the nonzero causal effects. The HPD and equal
tailed intervals are also very similar, suggesting relatively well
concentrated posterior distributions for the estimated effects in this
example.

## Cyclic (DCG) Example

We next illustrate `BayesDCG()`, the Bayesian sampler for directed
cyclic graph (DCG) structure estimation. We generate a DCG containing
directed feedback cycles under the disjoint-cycle restriction used by
`BayesDCG()`, so that no two directed cycles share a node.

For each included edge $j \to i$, the corresponding nonzero causal
effect coefficient is generated according to

$$|B_{ij}| \sim \mathrm{Unif}(0.4,0.9),$$

with its sign chosen independently to be positive or negative with equal
probability. For excluded edges, $B_{ij}=0$.

Because feedback is present in a cyclic model, we additionally require
the causal effect matrix to satisfy the stability condition

$$\rho(B) = \max\{|\lambda| : \lambda \in \mathrm{eig}(B)\} < 1.$$

Here, ρ(B) denotes the spectral radius of the causal effect matrix.
Given the causal effect matrix $B$ and structural error matrix
$\epsilon$, the observed data are generated from

$$Y = (I-B)^{-1}\epsilon.$$

The resulting data therefore arise from a stable linear non-Gaussian
cyclic structural equation model with disjoint directed cycles.

``` r
#######################################
# Simulation and MCMC settings
#######################################

N = 200
num_covariates = 10
M = 5
num_iter = 100000
burn_in_iterations = 80000


#######################################
# Hyperparameter setup
#######################################

params = list(
  a_mu      = 0,
  b_mu      = 2,
  a_gamma   = 1,
  b_gamma   = 20,
  a_gamma_1 = 0.5,
  b_gamma_1 = 0.5,
  a_tao     = 2,
  b_tao     = 1,
  alpha     = 1
)


#######################################
# Generate DCG Example
#######################################

example_list = generates_examples_DCG(
  num_covariates = num_covariates,   # Number of features
  N              = N,                # Sample size
  M_input        = 2,                # Number of Gaussian mixture components used to generate errors
  prob_sparsity  = 0.90,             # Probability of no edge (target edge density = 0.10)
  seed_input     = 21,               # Random seed for data generation
  n_cycles       = 2,                # Number of vertex-disjoint directed cycles
  len_range      = c(2, 4),          # Range of directed cycle lengths
  mag_range      = c(0.4, 0.9),      # Range of absolute values of nonzero edge weights
  prob_positive  = 0.5,              # Probability that a nonzero edge weight is positive
  tol_sing       = 0.05,             # Tolerance for near-singularity of the SEM system
  rho_max        = 0.95              # Maximum allowed spectral radius of the coefficient matrix
)


#######################################
# Extract generated data
#######################################

data_matrix = example_list$data_matrix
Adjacency_matrix_true = example_list$Adjacency_matrix_true
Causal_effect_matrix_true = example_list$Causal_effect_matrix_true
Z_matrix_true = example_list$Z_matrix_true

cycles_true = example_list$cycles
rho_true = example_list$rho


#######################################
# Inspect generated DCG
#######################################

cycles_true
#> [[1]]
#> [1] 3 9 2 7
#> 
#> [[2]]
#> [1]  5 10
rho_true
#> [1] 0.7454885
```

The generated DCG can be inspected using `cycles_true` and `rho_true`.
Here, `cycles_true` lists the directed cycles used to construct the
graph. For this example, the generated graph contains two vertex
disjoint cycles:

- `3 -> 9 -> 2 -> 7 -> 3`
- `5 -> 10 -> 5`

Thus, no node belongs to more than one directed cycle, consistent with
the disjoint cycle restriction used by `BayesDCG`.

The value `rho_true = 0.7454885` is the spectral radius of the generated
causal effect matrix. Since this value is below the specified stability
threshold (`rho_max = 0.95` in this example), the generated DCG
satisfies the stability requirement.

``` r
#######################################
# Parameter Initialization
#######################################

B_seed = directlingam_seed(data_matrix)

seed_edge_threshold = 0.05  # Threshold for removing small absolute edge weights
seed_rho_target = 0.95      # Target spectral radius for stabilizing the initial coefficient matrix
seed_settle_sweeps = 30     # Number of Gibbs sampling sweeps for mixture initialization

#######################################
# Initialization
#######################################

init_state = init_from_seed(
  B_seed, data_matrix, N, num_covariates, M,
  params$a_mu, params$b_mu,
  params$a_tao, params$b_tao,
  params$alpha,
  params$a_gamma, params$b_gamma,
  params$a_gamma_1, params$b_gamma_1,
  edge_threshold = seed_edge_threshold,
  rho_target     = seed_rho_target,
  settle_sweeps  = seed_settle_sweeps
)
#> init_from_seed: edges 8  rho 0.000  gamma_1 0.526  gamma_result 0.081  log-post -409.9


#######################################
# Run Bayesian DCG Sampler
#######################################

results_list = BayesDCG(
  data_matrix = data_matrix,

  a_mu      = params$a_mu,
  b_mu      = params$b_mu,
  a_gamma   = params$a_gamma,
  b_gamma   = params$b_gamma,
  a_tao     = params$a_tao,
  b_tao     = params$b_tao,
  a_gamma_1 = params$a_gamma_1,
  b_gamma_1 = params$b_gamma_1,
  alpha     = params$alpha,

  M        = M,
  num_iter = num_iter,

  burn_in_iterations = burn_in_iterations,

  init_Adjacency     = init_state$Adjacency_matrix,
  init_Causal_effect = init_state$Causal_effect_matrix,
  init_mu            = init_state$mu_mat,
  init_tao           = init_state$tao_mat,
  init_pi            = init_state$pi_mat,
  init_Z             = init_state$Z_matrix,
  init_gamma_1       = init_state$gamma_1,
  init_gamma_result  = init_state$gamma_result
)
#> [BayesDCG two-phase] 25%  (25000/100000)
#> [BayesDCG two-phase] 50%  (50000/100000)
#> [BayesDCG two-phase] 75%  (75000/100000)
#> [BayesDCG two-phase] done -- 100000 iterations


#######################################
# Extract posterior outputs
#######################################

log_likelihood_list        = results_list$log_likelihood_list
Adjacency_matrix_list      = results_list$Adjacency_matrix_list
Causal_effect_matrix_list  = results_list$Causal_effect_matrix_list
gamma_list                 = results_list$gamma_list
gamma_1_list               = results_list$gamma_1_list
mu_matrix_list             = results_list$mu_matrix_list
tao_matrix_list            = results_list$tao_matrix_list
pi_matrix_list             = results_list$pi_matrix_list

# Number of retained posterior graph samples
posterior_sample_length = nrow(Adjacency_matrix_list)
```

Before examining posterior summaries, again we will visualize the true
underlying DCG used in the simulation. The plot below displays the
ground truth adjacency structure.

``` r
#######################################
# Build directed graph from adjacency matrix
#######################################

g_true = graph_from_adjacency_matrix(
Adjacency_matrix_true,
mode = "directed",
diag = FALSE
)

plot(
  g_true,
  vertex.size = 20,
  vertex.label.cex = 0.8,
  edge.arrow.size = 0.5,
  main = "True DCG Structure"
)
```

<img src="man/figures/README-unnamed-chunk-16-1.png" alt="" width="100%" />

`BayesDCG()` returns the same general types of posterior output as
`BayesDAG()`, including sampled adjacency matrices, causal effect
matrices, and mixture model parameters.

As in the DAG example, we use `point_est_graph()` to obtain a
representative posterior graph. SHD and user defined distances may be
used for DCGs, SID is restricted to DAGs.

``` r
# SID is shown for completeness, it applies only when all posterior graphs are DAGs. If any sampled graph contains a cycle, SID-based selection will produce an error.
Adjacency_matrix_sid = point_est_graph(Adjacency_matrix_list, dist_type = 'sid')
#> Error in `sid_matrix()`:
#> ! SID distance requires all graphs to be DAGs.
```

``` r
Adjacency_matrix_shd = point_est_graph(Adjacency_matrix_list,dist_type = 'shd')
Adjacency_matrix_shd
#>       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#>  [1,]    0    0    1    1    0    0    0    0    1     0
#>  [2,]    0    0    0    0    0    0    0    0    1     0
#>  [3,]    0    0    0    0    0    0    1    0    0     0
#>  [4,]    0    0    0    0    0    0    0    0    0     0
#>  [5,]    0    0    0    0    0    0    0    0    0     1
#>  [6,]    0    0    0    0    0    0    0    0    0     0
#>  [7,]    0    1    0    0    0    0    0    0    0     0
#>  [8,]    0    0    0    0    0    0    0    0    0     0
#>  [9,]    0    0    1    0    0    0    0    0    0     0
#> [10,]    0    0    0    0    1    0    0    0    0     0
```

Using the same custom function as the acyclic case, we get following
graph structure.

``` r
Adjacency_matrix_custom = point_est_graph(Adjacency_matrix_list, dist_type = 'custom', dist_fun = custom_edge_mismatch)
Adjacency_matrix_custom
#>       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#>  [1,]    0    0    1    1    0    0    0    0    1     0
#>  [2,]    0    0    0    0    0    0    0    0    1     0
#>  [3,]    0    0    0    0    0    0    1    0    0     0
#>  [4,]    0    0    0    0    0    0    0    0    0     0
#>  [5,]    0    0    0    0    0    0    0    0    0     1
#>  [6,]    0    0    0    0    0    0    0    0    0     0
#>  [7,]    0    1    0    0    0    0    0    0    0     0
#>  [8,]    0    0    0    0    0    0    0    0    0     0
#>  [9,]    0    0    1    0    0    0    0    0    0     0
#> [10,]    0    0    0    0    1    0    0    0    0     0
```

In this example, the selected graphs coincide with the true adjacency
matrix:

``` r
Adjacency_matrix_true
#>       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#>  [1,]    0    0    1    1    0    0    0    0    1     0
#>  [2,]    0    0    0    0    0    0    0    0    1     0
#>  [3,]    0    0    0    0    0    0    1    0    0     0
#>  [4,]    0    0    0    0    0    0    0    0    0     0
#>  [5,]    0    0    0    0    0    0    0    0    0     1
#>  [6,]    0    0    0    0    0    0    0    0    0     0
#>  [7,]    0    1    0    0    0    0    0    0    0     0
#>  [8,]    0    0    0    0    0    0    0    0    0     0
#>  [9,]    0    0    1    0    0    0    0    0    0     0
#> [10,]    0    0    0    0    1    0    0    0    0     0
```

We can similarly use `posterior_network_motif()` to evaluate posterior
support for the true DCG motif:

``` r
true_graph_structure = igraph::graph_from_adjacency_matrix(Adjacency_matrix_true)

posterior_network_motif(true_graph_structure, Adjacency_matrix_list)
#> [1] 0.70445
```

The posterior probability of the true network motif is $0.704$,
indicating that the motif is present in approximately $70.4\%$ of the
retained posterior graph samples. This suggests substantial posterior
support for the underlying structural feature, even though the exact
full graph structure may vary across posterior draws.

We also inspect the retained log likelihood values to assess sampler
behavior:

``` r
plot(log_likelihood_list, type='l', xlab = 'Iterations', ylab = 'log likelihood values')
```

<img src="man/figures/README-unnamed-chunk-22-1.png" alt="" width="100%" />

The retained log likelihood remains within a relatively stable range,
providing empirical evidence that the sampler has entered a stable
sampling regime. We then use `posterior_interval_est()` to summarize
uncertainty in the causal effect coefficients, as in the DAG example.

``` r
#######################################
# Posterior intervals
#######################################

Causal_effect_matrix_summary =
  posterior_interval_est(
    Causal_effect_matrix_list,
    level = 0.95
  )

hpd_matrix_cyclic = Causal_effect_matrix_summary$hpd_matrix
ci_matrix_cyclic  = Causal_effect_matrix_summary$ci_matrix


#######################################
# True nonzero causal effects
#######################################

# R vectorizes matrices column-by-column
true_vec = as.vector(Causal_effect_matrix_true)

# Indices of all truly nonzero causal effects
true_idx = which(true_vec != 0)

# Convert vector indices back to matrix coordinates
coords = arrayInd(
  true_idx,
  .dim = dim(Causal_effect_matrix_true)
)

# Convention:
# B_ij is the causal effect j -> i
true_effects_df = data.frame(
  coef_index = true_idx,
  child  = coords[, 1],
  parent = coords[, 2],
  truth  = true_vec[true_idx]
)

true_effects_df$edge = paste0(
  true_effects_df$parent,
  " -> ",
  true_effects_df$child
)


#######################################
# Check dimensions before matching
#######################################

if (ncol(hpd_matrix_cyclic) != length(true_vec)) {
  stop(
    "HPD matrix ordering/dimensions do not match Causal_effect_matrix_true."
  )
}

if (nrow(ci_matrix_cyclic) != length(true_vec)) {
  stop(
    "CI matrix ordering/dimensions do not match Causal_effect_matrix_true."
  )
}


#######################################
# HPD intervals for TRUE causal effects
#######################################

hpd_sub = t(
  hpd_matrix_cyclic[
    ,
    true_idx,
    drop = FALSE
  ]
)

colnames(hpd_sub) = c(
  "lower",
  "upper"
)

data_hpd_cyclic = data.frame(
  edge  = true_effects_df$edge,
  truth = true_effects_df$truth,
  lower = hpd_sub[, "lower"],
  upper = hpd_sub[, "upper"]
)

# Midpoint of HPD interval
data_hpd_cyclic$estimate =
  (
    data_hpd_cyclic$lower +
    data_hpd_cyclic$upper
  ) / 2

# Keep edge order fixed
data_hpd_cyclic$edge =
  factor(
    data_hpd_cyclic$edge,
    levels = data_hpd_cyclic$edge
  )


#######################################
# Plot HPD intervals
#######################################

ggplot(
  data_hpd_cyclic,
  aes(
    x = edge,
    y = estimate
  )
) +

  # 95% HPD interval
  geom_errorbar(
    aes(
      ymin = lower,
      ymax = upper
    ),
    width = 0.2
  ) +

  # Posterior estimate
  geom_point(
    size = 3,
    color = "black"
  ) +

  # True causal effect: red X
  geom_point(
    aes(
      y = truth
    ),
    shape = 4,
    size = 4,
    stroke = 1.3,
    color = "red"
  ) +

  labs(
    x = "True Causal Edge",
    y = "Causal Effect",
    title = "Posterior Estimates with 95% HPD Intervals"
  ) +

  theme_minimal() +

  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    )
  )


#######################################
# 95% credible intervals for TRUE causal effects
#######################################

ci_sub =
  ci_matrix_cyclic[
    true_idx,
    ,
    drop = FALSE
  ]

data_ci_cyclic = data.frame(
  edge     = true_effects_df$edge,
  truth    = true_effects_df$truth,
  lower    = ci_sub[, 1],
  estimate = ci_sub[, 2],
  upper    = ci_sub[, 3]
)

# Keep edge order fixed
data_ci_cyclic$edge =
  factor(
    data_ci_cyclic$edge,
    levels = data_ci_cyclic$edge
  )


#######################################
# Plot credible intervals
#######################################

ggplot(
  data_ci_cyclic,
  aes(
    x = edge,
    y = estimate
  )
) +

  # 95% credible interval
  geom_errorbar(
    aes(
      ymin = lower,
      ymax = upper
    ),
    width = 0.2
  ) +

  # Posterior estimate
  geom_point(
    size = 3,
    color = "black"
  ) +

  # True causal effect: red X
  geom_point(
    aes(
      y = truth
    ),
    shape = 4,
    size = 4,
    stroke = 1.3,
    color = "red"
  ) +

  labs(
    x = "True Causal Edge",
    y = "Causal Effect",
    title = "Posterior Estimates with 95% Credible Intervals"
  ) +

  theme_minimal() +

  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    )
  )


#######################################
# Coverage
#######################################

data_hpd_cyclic$covered =
  data_hpd_cyclic$truth >= data_hpd_cyclic$lower &
  data_hpd_cyclic$truth <= data_hpd_cyclic$upper

data_ci_cyclic$covered =
  data_ci_cyclic$truth >= data_ci_cyclic$lower &
  data_ci_cyclic$truth <= data_ci_cyclic$upper


cat(
  "HPD coverage:",
  sum(data_hpd_cyclic$covered),
  "out of",
  nrow(data_hpd_cyclic),
  "\n"
)
#> HPD coverage: 7 out of 9

cat(
  "Credible interval coverage:",
  sum(data_ci_cyclic$covered),
  "out of",
  nrow(data_ci_cyclic),
  "\n"
)
#> Credible interval coverage: 7 out of 9
```

<img src="man/figures/README-unnamed-chunk-23-1.png" alt="" width="100%" /><img src="man/figures/README-unnamed-chunk-23-2.png" alt="" width="100%" />
The figures compare posterior estimates of the nonzero causal effect
coefficients with their true simulated values. Black points and vertical
lines represent the posterior estimates and corresponding 95% intervals,
while red crosses indicate the true causal effects.

For most edges, the posterior estimates are close to the true values and
the intervals contain or closely approach the corresponding true
effects. The largest discrepancies occur for the reciprocal edges
`5 -> 10` and `10 -> 5`, which form a two-node directed cycle in the
true graph. In particular, the posterior provides relatively weak
support for the `10 -> 5` effect, resulting in substantial posterior
mass near zero and an interval that does not contain the true positive
coefficient. The `5 -> 10` effect is retained but is overestimated in
this example.

These results illustrate that uncertainty in the causal effect estimates
reflects both uncertainty in graph selection and uncertainty in the
magnitude of an included edge. In particular, coefficients are equal to
zero in posterior draws for which the corresponding edge is absent.

## Documentation (vignette + appendix)

For a complete tutorial **and the supplemental material** (appendix with
additional simulations and implementation details), see the package
vignette:

``` r
#browseVignettes("cyclinbayes")
```
