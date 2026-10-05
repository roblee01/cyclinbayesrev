#' Bayesian DAG Causal Discovery
#'
#' @description
#' Fits a Bayesian linear non-Gaussian structural equation model under the
#' assumption that the causal graph is a directed acyclic graph (DAG).
#' Non-Gaussian errors are modeled using a finite normal mixture, and the
#' function returns posterior samples of the graph structure, causal effect
#' coefficients, mixture parameters, and sparsity parameters.
#'
#' @details
#' \code{BayesDAG} uses a collapsed Gibbs sampler for graph structure learning,
#' with an annealing period during the early iterations to improve exploration.
#' Only post-burn-in draws are returned. If \code{burn_in_iterations} is smaller
#' than 20 percent of \code{num_iter}, the effective burn-in is automatically
#' increased to the end of the annealing period.
#'
#' @param data_matrix Numeric matrix of dimension \eqn{N \times p}, where rows
#'   correspond to observations and columns correspond to variables.
#' @param a_mu Numeric. Mean of the normal prior on each mixture component mean.
#'   Default is 0.
#' @param b_mu Numeric. Variance of the normal prior on each mixture component
#'   mean. Default is 2.
#' @param a_gamma Numeric. First shape parameter of the Beta prior on the
#'   edge-inclusion probability \eqn{\gamma}. Default is 0.5.
#' @param b_gamma Numeric. Second shape parameter of the Beta prior on
#'   \eqn{\gamma}. Default is 0.5.
#' @param a_tao Numeric. Shape parameter of the inverse-gamma prior on the
#'   mixture component variances. Default is 2.
#' @param b_tao Numeric. Scale parameter of the inverse-gamma prior on the
#'   mixture component variances. Default is 1.
#' @param a_og_tao Numeric. Shape parameter used in the variance related proposal
#'   distribution. Default is 0.01.
#' @param b_og_tao Numeric. Scale parameter used in the variance related proposal
#'   distribution. Default is 0.01.
#' @param a_gamma_1 Numeric. Shape parameter of the inverse-gamma prior on the
#'   slab variance \eqn{\gamma_1}. Default is 2.
#' @param b_gamma_1 Numeric. Scale parameter of the inverse-gamma prior on
#'   \eqn{\gamma_1}. Default is 1.
#' @param alpha Numeric. Concentration parameter of the Dirichlet prior on the
#'   mixture weights. Default is 1.
#' @param M Integer. Maximum number of mixture components in the finite normal
#'   mixture error model.
#' @param num_iter Integer. Total number of MCMC iterations. Default is 10000.
#' @param burn_in_iterations Integer. Requested number of initial MCMC iterations
#'   to discard. The effective burn-in is at least 20 percent of
#'   \code{num_iter}. Default is 2000.
#' @param init_Adjacency Optional \eqn{p \times p} starting adjacency matrix.
#'   Entry \eqn{(i,j)} is 1 when node \eqn{j} is a parent of node \eqn{i}.
#'   The diagonal is ignored and the initial graph must be acyclic.
#'   Defaults to \code{NULL}.
#' @param init_Causal_effect Optional \eqn{p \times p} matrix of starting causal
#'   effect coefficients. Entries corresponding to absent edges are set to zero.
#'   Defaults to \code{NULL}.
#' @param init_mu Optional \eqn{p \times M} matrix of starting mixture component
#'   means. Defaults to \code{NULL}.
#' @param init_tao Optional \eqn{p \times M} matrix of starting mixture component
#'   variances. Defaults to \code{NULL}.
#' @param init_pi Optional \eqn{p \times M} matrix of starting mixture weights.
#'   Each row should sum to 1. Defaults to \code{NULL}.
#' @param init_Z Optional \eqn{(pN) \times M} indicator matrix of starting
#'   mixture memberships, stacked variable by variable, with one 1 per row.
#'   Defaults to \code{NULL}.
#' @param init_gamma_1 Optional numeric starting value for the slab variance
#'   \eqn{\gamma_1}. Defaults to \code{NULL}.
#' @param init_gamma_result Optional numeric starting value for the edge-inclusion
#'   probability \eqn{\gamma}. Defaults to \code{NULL}.
#'
#' @return A list containing retained post-burn-in posterior draws:
#' \describe{
#'   \item{Adjacency_matrix_list}{Vectorized draws of the \eqn{p \times p}
#'     adjacency matrix, one row per retained posterior draw.}
#'   \item{Causal_effect_matrix_list}{Vectorized draws of the \eqn{p \times p}
#'     causal effect matrix, one row per retained posterior draw.}
#'   \item{gamma_list}{Retained draws of the edge-inclusion probability
#'     \eqn{\gamma}.}
#'   \item{gamma_1_list}{Retained draws of the slab variance \eqn{\gamma_1}.}
#'   \item{mu_matrix_list}{Retained draws of the mixture component means.}
#'   \item{tao_matrix_list}{Retained draws of the mixture component variances.}
#'   \item{pi_matrix_list}{Retained draws of the mixture weights.}
#'   \item{log_likelihood_list}{Log-likelihood values for the retained posterior
#'     draws.}
#'   \item{first_stored_iteration}{Original MCMC iteration corresponding to the
#'     first returned draw.}
#'   \item{n_stored}{Number of retained posterior draws.}
#' }
#'
#' @export
#'
#' @examples
#' # Generate a DAG example
#' N <- 200
#' num_covariates <- 10
#' M <- 5
#' num_iter <- 5000
#' burn_in_iterations <- 1000
#'
#' example_list <- generates_examples_DAG(
#'   num_covariates = num_covariates,
#'   N = N, M_input = 2, prob_sparsity = 0.90,
#'   seed_input = 21, mag_range = c(0.4, 0.9),
#'   prob_positive = 0.5, seed_structure = 1, seed_weights = 20
#' )
#'
#' data_matrix <- example_list$data_matrix
#' Adjacency_matrix_true <- example_list$Adjacency_matrix_true
#'
#' # Fit the Bayesian DAG model
#' results_list <- BayesDAG(
#'   data_matrix,
#'   a_mu = 0, b_mu = 2,
#'   a_gamma = 0.5, b_gamma = 0.5,
#'   a_tao = 2, b_tao = 1,
#'   a_og_tao = 0.01, b_og_tao = 0.01,
#'   a_gamma_1 = 2, b_gamma_1 = 1,
#'   alpha = 1, M = M, num_iter = num_iter,
#'   burn_in_iterations = burn_in_iterations
#' )
#'
#' Adjacency_matrix_est <- point_est_graph(
#'   results_list$Adjacency_matrix_list,
#'   dist_type = "shd"
#' )
#'
#' Adjacency_matrix_true
#' Adjacency_matrix_est
#'
#' PIP_matrix <- matrix(
#'   colMeans(results_list$Adjacency_matrix_list),
#'   nrow = num_covariates,
#'   ncol = num_covariates
#' )
#'
#' PIP_matrix
#' head(results_list$log_likelihood_list)
#'
BayesDAG <- function(data_matrix, a_mu = 0, b_mu = 2, a_gamma = 0.5, b_gamma = 0.5,
                     a_tao = 2, b_tao = 1, a_og_tao = 0.01, b_og_tao = 0.01,
                     a_gamma_1 = 2, b_gamma_1 = 1, alpha = 1, M, num_iter = 10000,
                     burn_in_iterations = 2000,
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



