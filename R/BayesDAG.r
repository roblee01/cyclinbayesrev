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
#' @param num_iter Integer giving the total number of MCMC iterations for the \code{BayesDAG} algorithm.
#'
#' @return A list (from the C backend) typically containing adjacency and causal effect matrices and, optionally, samples/diagnostics.
#'
#' @export
#' @examples
#' # Run BayesDAG based on simulated example
#'
#' set.seed(21)
#'
#' # Simulation Settings
#'
#' N = 300 # sample size
#' num_covariates = 10 # number of variables (p)
#' M = 2 # number of Gaussian mixture components
#' num_iter = 5000 # number of MCMC iterations
#'
#'
#' # Generate Synthetic DAG Data
#'
#' example_list = generates_examples_DAG(num_covariates, N, M, 0.9, 21)
#'
#' data_matrix = example_list$data_matrix # generated data
#' Adjacency_matrix_true = example_list$Adjacency_matrix_true # true graph structure
#'
#'
#' # Hyperparameter Structure
#'
#' params = list(
#' a_mu = 0,
#' b_mu = 2,
#' a_gamma = 0.5,
#' b_gamma = 0.5,
#' a_gamma_1 = 2,
#' b_gamma_1 = 1,
#' a_tao = 2,
#' b_tao = 1,
#' a_og_tao = 0.01,
#' b_og_tao = 0.01,
#' alpha = 1
#' )
#'
#' # Run Bayesian LiNGAM (DAG) with a small iteration count
#'
#' results_lists = BayesDAG(
#' data_matrix,
#' params$a_mu,
#' params$b_mu,
#' params$a_gamma,
#' params$b_gamma,
#' params$a_tao,
#' params$b_tao,
#' params$a_og_tao,
#' params$b_og_tao,
#' params$a_gamma_1,
#' params$b_gamma_1,
#' params$alpha,
#' M,
#' num_iter
#' )
#'
#' # Basic posterior summaries
#'
#' Adjacency_matrix_list = results_lists$Adjacency_matrix_list
#' Causal_effect_matrix_list = results_lists$Causal_effect_matrix_list
#' gamma_list = results_lists$gamma_list
#' gamma_1_list = results_lists$gamma_1_list
#' mu_matrix_list = results_lists$mu_matrix_list
#' tao_matrix_list = results_lists$tao_matrix_list
#' pi_matrix_list = results_lists$pi_matrix_list
#' log_likelihood_list = results_lists$log_likelihood_list
#'
#' head(log_likelihood_list)

BayesDAG <- function(data_matrix, a_mu = 0, b_mu = 2, a_gamma = 0.5, b_gamma = 0.5, a_tao = 2, b_tao = 1, a_og_tao = 0.01, b_og_tao = 0.01, a_gamma_1 = 2, b_gamma_1 = 1, alpha = 1, M, num_iter) {
  return(BayesSCLingam_cpp(data_matrix, a_mu, b_mu, a_gamma, b_gamma, a_tao, b_tao, a_og_tao, b_og_tao, a_gamma_1, b_gamma_1, alpha, M, num_iter))
  #.Call('_cyclinbayes_BayesSCLingam', PACKAGE = 'cyclinbayes', data_matrix, a_mu, b_mu, a_gamma, b_gamma, a_tao, b_tao, a_og_tao, b_og_tao, a_gamma_1, b_gamma_1, alpha, M, num_iter)
}



