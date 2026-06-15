#' Compute Posterior Network Motif for Given Graph Structure
#'
#' @description
#' This function computes the posterior mass (relative frequency) of a specified graph structure among the collection of
#' sampled posterior adjacency matrices. The input graph is converted into an adjacency matrix vector, and function
#' counts how often all edges appear in sampled posterior graph.
#'
#'
#' @param graph_structure An \code{igraph} object representing the target graph.
#' @param posterior_graph_structures A matrix of dimension \code{num_iter × p^2}, where each row corresponds
#' to a flattened adjacency matrix from one MCMC iteration.
#' @importFrom igraph as_adjacency_matrix
#'
#' @return A numeric value between 0 and 1 giving the posterior mass (frequency) with which the target
#' graph's edges appear in the posterior samples.
#'
#' @export
#' @examples
#'
#' # Run BayesDAG simulated example
#'
#' N = 300
#' num_covariates = 10
#' M = 2
#' num_iter = 1000
#'
#'
#' example_list = generates_examples_DAG(num_covariates, N, M, 0.9, 21)
#'
#'
#' data_matrix = example_list$data_matrix
#' Adjacency_matrix_true = example_list$Adjacency_matrix_true
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
#'
#' result_list = BayesDAG(
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
#' ) # Runs the Acyclic algorithm
#'
#'
#' # Extract posterior graph structures
#' Adjacency_matrix_list = result_list$Adjacency_matrix_list
#'
#'
#' # Compute the posterior mass assigned to true graph
#'
#' true_graph_structure = igraph::graph_from_adjacency_matrix(Adjacency_matrix_true)
#' posterior_network_motif(true_graph_structure, Adjacency_matrix_list)

posterior_network_motif = function(graph_structure, posterior_graph_structures){
  A = as.matrix(as_adjacency_matrix(graph_structure))
  causal_effects_index = which(c(A) == 1)
  hits = apply(posterior_graph_structures, 1, function(row){
    all(causal_effects_index %in% which(row == 1))
  })
  return(sum(hits) / nrow(posterior_graph_structures))
}

