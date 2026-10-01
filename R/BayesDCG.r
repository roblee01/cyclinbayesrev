#' Bayesian Cyclic Causal Discovery
#'
#' @description
#' BayesDCG fits a Bayesian linear causal model on a directed cyclic graph (DCG) with non-Gaussian errors, returning posterior samples of the adjacency and causal-effect matrices for systems with feedback.
#'
#' @details
#' The sampler runs in two phases. Iterations up to \code{burn_in_iterations} are
#' unconstrained. The later iterations are restricted to graphs whose cycles are
#' vertex-disjoint, with a one-time projection at the handoff and an annealing
#' window immediately after it. Draws from the unconstrained and annealing
#' phases are not from the target posterior, so they are run but not stored:
#' every returned row is a posterior draw. Row \eqn{k} of each returned trace
#' corresponds to iteration \code{first_stored_iteration + k - 1}.
#'
#' @param data_matrix Numeric matrix of dimension \eqn{N \times p}, where rows correspond to observations and columns correspond to variables (features) included in the causal graph.
#' @param a_mu Hyperparameter for the mean of the normal prior on each mixture component mean \eqn{\mu_k} in the error mixture model (location parameter). Default value is 0.
#' @param b_mu Hyperparameter for the variance of the normal prior on each mixture component mean \eqn{\mu_k} (controls how tightly the component means are shrunk toward \code{a_mu}).
#' Default value is 2.
#' @param a_gamma First shape parameter of the Beta prior on the edge-inclusion probability \eqn{\gamma} (probability that there is an edge from one node to another).
#' Default value is 1.
#' @param b_gamma Second shape parameter of the Beta prior on the edge-inclusion probability \eqn{\gamma}. Along with \code{a_gamma} this controls the expected sparsity of the graph.
#' Default value is 20.
#' @param a_tao Shape parameter of the inverse gamma prior on each mixture component variance in the error distribution (controls the prior tail heaviness for component variances).
#' Default value is 2.
#' @param b_tao Scale parameter of the inverse gamma prior on each mixture component variance in the error distribution (sets the typical size of the component variances).
#' Default value is 1.
#' @param a_gamma_1 Shape parameter of the inverse gamma prior on the slab variance \eqn{\gamma_1} in the conditional spike-and-slab prior \eqn{B_{ij}\mid E_{ij},
#' \gamma_1 \sim (1-E_{ij})\delta_0 + E_{ij}N(0,\gamma_1)}. Default value is 0.5.
#' @param b_gamma_1 Scale parameter of the inverse gamma prior on \eqn{\gamma_1}. Default value is 0.5.
#' @param alpha Concentration parameter of the Dirichlet prior on the mixture weights for the normal mixture error distribution (controls how evenly the mixture components are used).
#' Default value is 1
#' @param M Integer giving the maximum number of mixture components allowed in the normal mixture error model.
#' @param num_iter Integer giving the total number of MCMC iterations for the \code{BayesDCG} algorithm.
#' @param burn_in_iterations Integer giving the length of the unconstrained phase. Must be smaller than \code{num_iter}, otherwise the constrained phase never runs and no draws are stored. Defaults to 70 percent of \code{num_iter}.
#' @param init_Adjacency Optional \eqn{p \times p} matrix giving the starting adjacency matrix. Entry \eqn{(i,j)} is 1 when \eqn{j} is a parent of \eqn{i}. The diagonal is ignored. Defaults to NULL, which starts from the prior/random draw.
#' @param init_Causal_effect Optional \eqn{p \times p} matrix of starting coefficients. Entries where \code{init_Adjacency} is 0 are set to 0. Defaults to NULL.
#' @param init_mu Optional \eqn{p \times M} matrix of starting mixture-component means. Defaults to NULL.
#' @param init_tao Optional \eqn{p \times M} matrix of starting mixture-component variances. Defaults to NULL.
#' @param init_pi Optional \eqn{p \times M} matrix of starting mixture weights; rows should sum to 1. Defaults to NULL.
#' @param init_Z Optional \eqn{(p N) \times M} indicator matrix of starting component memberships, stacked variable by variable, with one 1 per row. Defaults to NULL.
#' @param init_gamma_1 Optional starting value for the slab variance \eqn{\gamma_1}. Defaults to NULL.
#' @param init_gamma_result Optional starting value for the edge-inclusion probability \eqn{\gamma}. Defaults to NULL.
#'
#' @return A list with components
#' \describe{
#'   \item{Adjacency_matrix_list}{matrix of stored draws, one row per draw, each row \code{as.vector()} of the \eqn{p \times p} adjacency matrix}
#'   \item{Causal_effect_matrix_list}{stored draws of the coefficient matrix, same layout}
#'   \item{gamma_list, gamma_1_list}{stored draws of the edge-inclusion probability and the slab variance}
#'   \item{mu_matrix_list, tao_matrix_list, pi_matrix_list}{stored draws of the error-mixture means, variances and weights}
#'   \item{first_stored_iteration}{iteration number corresponding to the first stored row}
#'   \item{n_stored}{number of stored draws}
#' }
#'
#' @export
#' @examples
#' # Run BayesDCG on the DCG example used in the README
#'
#' N <- 200
#' num_covariates <- 10
#' M <- 5
#' num_iter <- 10000
#' burn_in_iterations <- 5000
#'
#' # Generate the same DCG example used in the README
#' example_list <- generates_examples_DCG(
#'   num_covariates = num_covariates,
#'   N              = N,
#'   M_input        = 2,
#'   prob_sparsity  = 0.90,
#'   seed_input     = 21,
#'   n_cycles       = 2,
#'   len_range      = c(2, 4),
#'   mag_range      = c(0.4, 0.9),
#'   prob_positive  = 0.5,
#'   tol_sing       = 0.05,
#'   rho_max        = 0.95
#' )
#'
#' data_matrix <- example_list$data_matrix
#' Adjacency_matrix_true <- example_list$Adjacency_matrix_true
#' Causal_effect_matrix_true <- example_list$Causal_effect_matrix_true
#' Z_matrix_true <- example_list$Z_matrix_true
#'
#' cycles_true <- example_list$cycles
#' rho_true <- example_list$rho
#'
#' # Inspect the generated cyclic structure
#' cycles_true
#' rho_true
#'
#' # Fit the Bayesian DCG model
#' results_list <- BayesDCG(
#'   data_matrix,
#'   a_mu = 0,
#'   b_mu = 2,
#'   a_gamma = 1,
#'   b_gamma = 20,
#'   a_tao = 2,
#'   b_tao = 1,
#'   a_gamma_1 = 0.5,
#'   b_gamma_1 = 0.5,
#'   alpha = 1,
#'   M = M,
#'   num_iter = num_iter,
#'   burn_in_iterations = burn_in_iterations
#' )
#'
#' # Number of retained posterior draws
#' results_list$n_stored
#'
#' # Select a representative posterior graph using
#' # posterior expected Structural Hamming Distance
#' Adjacency_matrix_est <- point_est_graph(
#'   results_list$Adjacency_matrix_list,
#'   dist_type = "shd"
#' )
#'
#' # Compare the true and estimated graph structures
#' Adjacency_matrix_true
#' Adjacency_matrix_est
#'
#' # Posterior edge-inclusion probabilities
#' PIP_matrix <- matrix(
#'   colMeans(results_list$Adjacency_matrix_list),
#'   nrow = num_covariates,
#'   ncol = num_covariates
#' )
#'
#' PIP_matrix
#'
#' # Inspect retained log-likelihood values
#' head(results_list$log_likelihood_list)

BayesDCG <- function(data_matrix, a_mu = 0, b_mu = 2, a_gamma = 1, b_gamma = 20,
                     a_tao = 2, b_tao = 1, a_gamma_1 = 0.5, b_gamma_1 = 0.5,
                     alpha = 1, M, num_iter,
                     burn_in_iterations = floor(0.7 * num_iter),
                     init_Adjacency = NULL, init_Causal_effect = NULL,
                     init_mu = NULL, init_tao = NULL, init_pi = NULL,
                     init_Z = NULL, init_gamma_1 = NULL,
                     init_gamma_result = NULL) {
  if (burn_in_iterations >= num_iter)
    stop("burn_in_iterations must be smaller than num_iter, otherwise no posterior draws are stored.")

  BCD_v2_two_phase_cpp(
    data_matrix, a_mu, b_mu, a_gamma, b_gamma,
    a_tao, b_tao, a_gamma_1, b_gamma_1,
    alpha, M, num_iter, burn_in_iterations,
    init_Adjacency     = .init_matrix(init_Adjacency,     "init_Adjacency"),
    init_Causal_effect = .init_matrix(init_Causal_effect, "init_Causal_effect"),
    init_mu            = .init_matrix(init_mu,            "init_mu"),
    init_tao           = .init_matrix(init_tao,           "init_tao"),
    init_pi            = .init_matrix(init_pi,            "init_pi"),
    init_Z             = .init_matrix(init_Z,             "init_Z"),
    init_gamma_1       = .init_scalar(init_gamma_1,       "init_gamma_1"),
    init_gamma_result  = .init_scalar(init_gamma_result,  "init_gamma_result")
  )
}
