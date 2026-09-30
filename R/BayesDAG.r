#' Bayesian Lingam Causal Discovery
#'
#' @description
#' Fits a Bayesian Collapsed Gibbs sampler of the LiNGAM model with non-Gaussian errors modeled
#' via a finite normal mixture, returning posterior samples for the graph
#' structure and causal effect coefficients.
#'
#'
#' @param data_matrix Numeric matrix of dimension \eqn{N \times p}, where rows correspond to observations and columns correspond to variables (features) included in the causal graph.
#' @param a_mu Hyperparameter for the mean of the normal prior on each mixture component mean \eqn{\mu_k} in the error mixture model (location parameter). Default value is 0.
#' @param b_mu Hyperparameter for the variance of the normal prior on each mixture component mean \eqn{\mu_k} (controls how tightly the component means are shrunk toward \code{a_mu}).
#' Default value is 2.
#' @param a_gamma First shape parameter of the Beta prior on the edge-inclusion probability \eqn{\gamma} (probability that there is an edge from one node to another).
#' Default value is 0.5.
#' @param b_gamma Second shape parameter of the Beta prior on the edge-inclusion probability \eqn{\gamma}. Along with \code{a_gamma} this controls the expected sparsity of the graph.
#' Default value is 0.5.
#' @param a_tao Shape parameter of the inverse gamma prior on each mixture component variance in the error distribution (controls the prior tail heaviness for component variances).
#' Default value is 2.
#' @param b_tao Scale parameter of the inverse gamma prior on each mixture component variance in the error distribution (sets the typical size of the component variances).
#' Default value is 1.
#' @param a_gamma_1 Shape parameter of the inverse gamma prior on the slab variance \eqn{\gamma_1} in the conditional spike and slab prior \eqn{B_{ij}\mid E_{ij},
#' \gamma_1 \sim (1-E_{ij})\delta_0 + E_{ij}N(0,\gamma_1)}. Default value is 2.
#' @param b_gamma_1 Scale parameter of the inverse gamma prior on \eqn{\gamma_1}. Default value is 1.
#' @param a_og_tao Shape parameter of the proposal distribution used when updating the variance parameters \eqn{\tau} associated with adjacency matrix entries in the MCMC algorithm.
#' Default value is 0.01.
#' @param b_og_tao Scale parameter of the proposal distribution used when updating the variance parameters \eqn{\tau} associated with adjacency matrix entries in the MCMC algorithm.
#' Default value is 0.01.
#' @param alpha Concentration parameter of the Dirichlet prior on the mixture weights for the normal mixture error distribution (controls how evenly the mixture components are used).
#' Default value is 1
#' @param M Integer giving the maximum number of mixture components allowed in the normal mixture error model.
#' @param init_Adjacency Optional \eqn{p \times p} matrix giving the starting adjacency matrix. Entry \eqn{(i,j)} is 1 when \eqn{j} is a parent of \eqn{i}. The diagonal is ignored. Must be acyclic, since this sampler's edge moves assume a DAG. Defaults to NULL, which starts from the prior/random draw.
#' @param init_Causal_effect Optional \eqn{p \times p} matrix of starting coefficients. Entries where \code{init_Adjacency} is 0 are set to 0. Defaults to NULL.
#' @param init_mu Optional \eqn{p \times M} matrix of starting mixture-component means. Defaults to NULL.
#' @param init_tao Optional \eqn{p \times M} matrix of starting mixture-component variances. Defaults to NULL.
#' @param init_pi Optional \eqn{p \times M} matrix of starting mixture weights; rows should sum to 1. Defaults to NULL.
#' @param init_Z Optional \eqn{(p N) \times M} indicator matrix of starting component memberships, stacked variable by variable, with one 1 per row. Defaults to NULL.
#' @param init_gamma_1 Optional starting value for the slab variance \eqn{\gamma_1}. Defaults to NULL.
#' @param init_gamma_result Optional starting value for the edge-inclusion probability \eqn{\gamma}. Defaults to NULL.
#' @param num_iter Integer giving the total number of MCMC iterations for the \code{BayesDAG} algorithm.
#' @param burn_in_iterations Integer giving the requested number of initial MCMC iterations to discard. Because \code{BayesDAG} uses an annealing period during early sampling, the effective burn-in is automatically increased to at least 20\% of \code{num_iter} when a smaller value is supplied. Only post-burn-in posterior draws are returned. Default value is 0.
#'
#' @return A list containing only retained post-burn-in posterior draws:
#' \describe{
#'   \item{Adjacency_matrix_list}{retained draws of the \eqn{p \times p} adjacency matrix, one \code{as.vector()} per row}
#'   \item{Causal_effect_matrix_list}{retained draws of the coefficient matrix, same layout}
#'   \item{gamma_list, gamma_1_list}{retained draws of the edge-inclusion probability and the slab variance}
#'   \item{mu_matrix_list, tao_matrix_list, pi_matrix_list}{retained draws of the error-mixture means, variances and weights}
#'   \item{log_likelihood_list}{log-likelihood values for the retained draws}
#'   \item{first_stored_iteration}{original MCMC iteration corresponding to the first returned draw}
#'   \item{n_stored}{number of retained posterior draws}
#' }
#'
#' @export
#' @examples
#' # Run BayesDAG on a simulated acyclic example
#'
#' set.seed(21)
#'
#' N <- 250               # sample size
#' num_covariates <- 7    # number of features
#' M <- 2                 # mixture components for the error distribution
#' num_iter <- 5000       # total MCMC iterations
#' burn_in_iterations <- 1000  # requested burn-in
#'
#' # True DAG and data
#' truth <- generate_dag(num_covariates, edge_prob = 0.15,
#'                       mag_range = c(0.4, 0.9))
#' Adjacency_matrix_true <- truth$E
#'
#' err <- matrix(rnorm(N * num_covariates, mean = 2 * sample(c(-1, 1),
#'               N * num_covariates, TRUE), sd = 0.5), N, num_covariates)
#' data_matrix <- t(solve(diag(num_covariates) - truth$B, t(err)))
#'
#' results_lists <- BayesDAG(
#'   data_matrix,
#'   a_mu = 0, b_mu = 2,
#'   a_gamma = 0.5, b_gamma = 0.5,
#'   a_tao = 2, b_tao = 1,
#'   a_og_tao = 0.01, b_og_tao = 0.01,
#'   a_gamma_1 = 2, b_gamma_1 = 1,
#'   alpha = 1,
#'   M = M,
#'   num_iter = num_iter,
#'   burn_in_iterations = burn_in_iterations
#' )
#'
#' # Returned traces already contain only retained posterior draws
#' results_lists$first_stored_iteration
#' results_lists$n_stored
#'
#' edge_prob <- matrix(colMeans(results_lists$Adjacency_matrix_list),
#'                     num_covariates, num_covariates)
#' estimated_graph <- (edge_prob > 0.5) * 1
#'
#' mean(estimated_graph == Adjacency_matrix_true)
#' head(results_lists$log_likelihood_list)
#'
#' # Continuing a chain: start a second run from the last draw.
#' # init_Adjacency must be acyclic, which any stored draw is.
#'
#' last <- nrow(results_lists$Adjacency_matrix_list)
#'
#' results_lists2 <- BayesDAG(
#'   data_matrix,
#'   M = M, num_iter = num_iter,
#'   burn_in_iterations = burn_in_iterations,
#'   init_Adjacency     = matrix(results_lists$Adjacency_matrix_list[last, ],
#'                               num_covariates, num_covariates),
#'   init_Causal_effect = matrix(results_lists$Causal_effect_matrix_list[last, ],
#'                               num_covariates, num_covariates),
#'   init_mu            = matrix(results_lists$mu_matrix_list[last, ], num_covariates, M),
#'   init_tao           = matrix(results_lists$tao_matrix_list[last, ], num_covariates, M),
#'   init_pi            = matrix(results_lists$pi_matrix_list[last, ], num_covariates, M),
#'   init_gamma_1       = results_lists$gamma_1_list[last],
#'   init_gamma_result  = results_lists$gamma_list[last]
#' )

BayesDAG <- function(data_matrix, a_mu = 0, b_mu = 2, a_gamma = 0.5, b_gamma = 0.5,
                     a_tao = 2, b_tao = 1, a_og_tao = 0.01, b_og_tao = 0.01,
                     a_gamma_1 = 2, b_gamma_1 = 1, alpha = 1, M, num_iter,
                     burn_in_iterations = 0,
                     init_Adjacency = NULL, init_Causal_effect = NULL,
                     init_mu = NULL, init_tao = NULL, init_pi = NULL,
                     init_Z = NULL, init_gamma_1 = NULL,
                     init_gamma_result = NULL) {
  return(BayesSCLingam_cpp(
    data_matrix, a_mu, b_mu, a_gamma, b_gamma, a_tao, b_tao,
    a_og_tao, b_og_tao, a_gamma_1, b_gamma_1, alpha, M, num_iter,
    burn_in_iterations = burn_in_iterations,
    init_Adjacency     = .init_matrix(init_Adjacency,     "init_Adjacency"),
    init_Causal_effect = .init_matrix(init_Causal_effect, "init_Causal_effect"),
    init_mu            = .init_matrix(init_mu,            "init_mu"),
    init_tao           = .init_matrix(init_tao,           "init_tao"),
    init_pi            = .init_matrix(init_pi,            "init_pi"),
    init_Z             = .init_matrix(init_Z,             "init_Z"),
    init_gamma_1       = .init_scalar(init_gamma_1,       "init_gamma_1"),
    init_gamma_result  = .init_scalar(init_gamma_result,  "init_gamma_result")
  ))
  #.Call('_cyclinbayes_BayesSCLingam', PACKAGE = 'cyclinbayes', data_matrix, a_mu, b_mu, a_gamma, b_gamma, a_tao, b_tao, a_og_tao, b_og_tao, a_gamma_1, b_gamma_1, alpha, M, num_iter)
}



