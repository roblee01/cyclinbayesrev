#' Select a point estimate graph from posterior samples
#'
#' @description
#' Selects a representative graph structure from posterior samples by computing a **weighted medoid** under a chosen distance. Intended for lists of sampled adjacency matrices
#'
#' @param Adjacency_matrix_list A matrix of dimension \code{num_iter × (p*p)} where each row is a flattened adjacency matrix sampled during the posterior (0/1 entries).
#' @param dist_type Character string specifying the distance metric.
#'   One of \code{"shd"}, \code{"sid"}, or \code{"custom"}.
#' @param dist_fun Optional user-supplied distance function used when
#'   \code{dist_type = "custom"}. Must have signature \code{dist_fun(A, B)},
#'   where \code{A} and \code{B} are \eqn{p \times p} adjacency matrices and
#'   return a non negative scalar distance.
#' @param burn_in_frac Fraction of iterations to discard as burn in (default 0.75).
#' @return The best possible graph structure found through finding the smallest distance through finding the weighted medoid based on chosen distance.
#'
#' @export
#' @examples
#' # This example runs the BayesDAG sampler and then selects a point-estimate graph.
#' # NOTE: If you want to use dist_type = "sid", you may need extra dependencies.
#' # In particular, on some systems the SID package requires Bioconductor RBGL/graph.
#'
#' \donttest{
#' # Install required packages if needed:
#' # install.packages(c("remotes", "SID"))
#' #
#' # If SID complains about missing RBGL/graph, install via Bioconductor:
#' # install.packages("BiocManager")
#' # BiocManager::install(c("graph", "RBGL"))
#' }
#'
#' N = 300
#' num_covariates = 10
#' M = 2
#' num_iter = 1000
#'
#' example_list = generates_examples_DAG(num_covariates, N, M, 0.9, 21)
#' data_matrix = example_list$data_matrix
#'
#' params = list(
#'   a_mu = 0, b_mu = 2,
#'   a_gamma = 0.5, b_gamma = 0.5,
#'   a_gamma_1 = 2, b_gamma_1 = 1,
#'   a_tao = 2, b_tao = 1,
#'   a_og_tao = 0.01, b_og_tao = 0.01,
#'   alpha = 1
#' )
#'
#' result_list = BayesDAG(
#'   data_matrix,
#'   params$a_mu, params$b_mu,
#'   params$a_gamma, params$b_gamma,
#'   params$a_tao, params$b_tao,
#'   params$a_og_tao, params$b_og_tao,
#'   params$a_gamma_1, params$b_gamma_1,
#'   params$alpha,
#'   M, num_iter
#' )
#'
#' Adjacency_matrix_list <- result_list$Adjacency_matrix_list
#'
#' # Best graph structure using SHD
#' point_est_graph(Adjacency_matrix_list, dist_type = "shd")
#'
#' # Best graph structure using SID (requires the SID package and possibly RBGL/graph)
#' if (requireNamespace("SID", quietly = TRUE)) {
#'   point_est_graph(Adjacency_matrix_list, dist_type = "sid")
#' }
#'
#' # Best graph structure using a custom distance
#' custom_edge_mismatch = function(A, B) sum(abs(A - B))
#' point_est_graph(Adjacency_matrix_list, dist_type = "custom", dist_fun = custom_edge_mismatch)



point_est_graph = function(Adjacency_matrix_list, dist_type = 'shd', dist_fun = NULL, burn_in_frac = 0.75){
  num_covariates = sqrt(ncol(Adjacency_matrix_list))
  num_iter = nrow(Adjacency_matrix_list)
  p = sqrt(ncol(Adjacency_matrix_list))


  start = floor(burn_in_frac * num_iter) + 1L
  keep_inds = start:num_iter
  A_keep = Adjacency_matrix_list[keep_inds, , drop = FALSE]





  structure_strings = apply(A_keep, 1, paste, collapse = ",")
  unique_strings = unique(structure_strings)

  tab = table(factor(structure_strings, levels = unique_strings))
  weights = as.numeric(tab)

  unique_graphs = lapply(unique_strings, function(s) {
    vec = as.numeric(strsplit(s, ",")[[1]])
    matrix(vec, p, p)
  })

  v = length(unique_graphs)

  D = matrix(0, nrow = v, ncol = v)

  if (dist_type == "shd") {

    for (i in seq_len(v - 1L)) {
      Ai = unique_graphs[[i]]
      for (j in (i + 1L):v) {
        Aj = unique_graphs[[j]]
        d_ij = sum(Ai != Aj)   # structural Hamming distance on adjacency matrices
        D[i, j] = d_ij
        D[j, i] = d_ij
      }
    }

  } else if(dist_type == 'sid'){

    is_dag_adj <- function(A) {
      A <- (A != 0) * 1L
      diag(A) <- 0L
      p <- nrow(A)

      indeg <- colSums(A)
      queue <- which(indeg == 0L)
      removed <- 0L

      while (length(queue) > 0L) {
        vtx <- queue[1L]
        queue <- queue[-1L]
        removed <- removed + 1L

        out <- which(A[vtx, ] != 0L)
        if (length(out)) {
          indeg[out] <- indeg[out] - 1L
          A[vtx, out] <- 0L
          queue <- c(queue, out[indeg[out] == 0L])
        }
      }
      removed == p
    }

    is_dag <- vapply(unique_graphs, is_dag_adj, logical(1L))
    if (!all(is_dag)) stop("SID distance requires all graphs to be DAGs.")

    for (i in seq_len(v - 1L)) {
      Ai <- (unique_graphs[[i]] != 0) * 1L
      diag(Ai) <- 0L
      for (j in (i + 1L):v) {
        Aj <- (unique_graphs[[j]] != 0) * 1L
        diag(Aj) <- 0L
        d_ij <- SID::structIntervDist(Ai, Aj)$sid
        D[i, j] <- d_ij
        D[j, i] <- d_ij
      }
    }
  } else if (dist_type == "custom") {
    if (is.null(dist_fun)) {
      stop("dist_type = 'custom' requires a user-supplied dist_fun(A, B).")
    }

    for (i in seq_len(v - 1L)) {
      Ai <- unique_graphs[[i]]
      for (j in (i + 1L):v) {
        d_ij <- dist_fun(Ai, unique_graphs[[j]])
        D[i, j] <- d_ij
        D[j, i] <- d_ij
      }
    }
  } else {
    stop("Unknown dist_type: ", dist_type)
  }

  total_distance = as.vector(D %*% weights)
  best_index = which.min(total_distance)

  best_adjacency_matrix = unique_graphs[[best_index]]
  return(best_adjacency_matrix)
}
