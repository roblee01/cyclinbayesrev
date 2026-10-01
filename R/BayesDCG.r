#' Bayesian Cyclic Causal Discovery
#'
#' @description
#' BayesDCG fits a Bayesian linear causal model on a directed cyclic graph (DCG) with non-Gaussian errors, returning posterior samples of the adjacency and causal-effect matrices for systems with feedback.
#'
#' @details
#' The sampler runs in two phases. Iterations up to \code{burn_in_iterations} are
#' unconstrained; later iterations are restricted to graphs whose cycles are
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
#' # Run BayesDCG on a simulated cyclic example
#'
#' set.seed(21)
#'
#' N <- 250               # sample size
#' num_covariates <- 7    # number of features
#' M <- 2                 # mixture components for the error distribution
#' num_iter <- 5000       # MCMC iterations
#'
#' # True cyclic graph and data
#' truth <- generates_examples_DCG(num_covariates, edge_prob = 0.15, n_cycles = 2,
#'                       mag_range = c(0.4, 0.9))
#' Adjacency_matrix_true <- truth$E
#'
#' err <- matrix(rnorm(N * num_covariates, mean = 2 * sample(c(-1, 1),
#'               N * num_covariates, TRUE), sd = 0.5), N, num_covariates)
#' data_matrix <- t(solve(diag(num_covariates) - truth$B, t(err)))
#'
#' result_list <- BayesDCG(
#'   data_matrix,
#'   a_mu = 0, b_mu = 2,
#'   a_gamma = 1, b_gamma = 20,
#'   a_tao = 2, b_tao = 1,
#'   a_gamma_1 = 0.5, b_gamma_1 = 0.5,
#'   alpha = 1,
#'   M = M,
#'   num_iter = num_iter,
#'   burn_in_iterations = 3500
#' )
#'
#' # Posterior edge probabilities and a thresholded graph
#' edge_prob <- matrix(colMeans(result_list$Adjacency_matrix_list),
#'                     num_covariates, num_covariates)
#' estimated_graph <- (edge_prob > 0.5) * 1
#'
#' result_list$n_stored
#' mean(estimated_graph == Adjacency_matrix_true)
#'
#' # Continuing a chain: start a second run from the last stored draw
#'
#' last <- result_list$n_stored
#'
#' result_list2 <- BayesDCG(
#'   data_matrix,
#'   M = M, num_iter = num_iter, burn_in_iterations = 3500,
#'   init_Adjacency     = matrix(result_list$Adjacency_matrix_list[last, ],
#'                               num_covariates, num_covariates),
#'   init_Causal_effect = matrix(result_list$Causal_effect_matrix_list[last, ],
#'                               num_covariates, num_covariates),
#'   init_mu            = matrix(result_list$mu_matrix_list[last, ], num_covariates, M),
#'   init_tao           = matrix(result_list$tao_matrix_list[last, ], num_covariates, M),
#'   init_pi            = matrix(result_list$pi_matrix_list[last, ], num_covariates, M),
#'   init_gamma_1       = result_list$gamma_1_list[last],
#'   init_gamma_result  = result_list$gamma_list[last]
#' )

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
