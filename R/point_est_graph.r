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
#'   return a non negative scalar distance. It is assumed to be symmetric.
#' @param burn_in_frac Fraction of iterations to discard as burn in (default 0.75).
#' @return The best possible graph structure found through finding the smallest distance through finding the weighted medoid based on chosen distance.
#'
#' @details
#' SID is not symmetric. For \code{dist_type = "sid"} each candidate graph is
#' scored as the estimate against every posterior sample as the truth, i.e. the
#' returned graph minimises \eqn{\sum_g w_g \, SID(G_g, \hat G)}, the posterior
#' expected SID. SID is computed natively (no dependency on the SID package).
#' It reproduces \code{SID::structIntervDist(trueGraph, estGraph)$sid} for DAGs.
#'
#' @export
#' @examples
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
#' # Best graph structure using SID
#' point_est_graph(Adjacency_matrix_list, dist_type = "sid")
#'
#' # Best graph structure using a custom distance
#' custom_edge_mismatch = function(A, B) sum(abs(A - B))
#' point_est_graph(Adjacency_matrix_list, dist_type = "custom", dist_fun = custom_edge_mismatch)

point_est_graph = function(Adjacency_matrix_list, dist_type = 'shd', dist_fun = NULL, burn_in_frac = 0.75){
  dist_type = match.arg(dist_type, c("shd", "sid", "custom"))

  if (!is.numeric(burn_in_frac) || length(burn_in_frac) != 1L ||
      is.na(burn_in_frac) || burn_in_frac < 0 || burn_in_frac >= 1) {
    stop("burn_in_frac must be a single numeric value in [0, 1).")
  }

  if (!is.matrix(Adjacency_matrix_list)) {
    Adjacency_matrix_list = as.matrix(Adjacency_matrix_list)
  }
  if (nrow(Adjacency_matrix_list) < 1L || ncol(Adjacency_matrix_list) < 1L) {
    stop("Adjacency_matrix_list must contain at least one posterior graph.")
  }

  num_iter = nrow(Adjacency_matrix_list)
  p = as.integer(round(sqrt(ncol(Adjacency_matrix_list))))
  if (p * p != ncol(Adjacency_matrix_list)) {
    stop("The number of columns in Adjacency_matrix_list must equal p^2 for some integer p.")
  }

  start = floor(burn_in_frac * num_iter) + 1L
  A_keep = Adjacency_matrix_list[start:num_iter, , drop = FALSE]

  # Unique graphs and their counts, without the paste/strsplit round trip
  keys = do.call(paste, c(as.data.frame(A_keep), sep = ""))
  first = !duplicated(keys)
  weights = tabulate(match(keys, keys[first]), nbins = sum(first))
  U = A_keep[first, , drop = FALSE]          # v x p^2, one unique graph per row
  v = nrow(U)

  if (v == 1L) return(matrix(U[1, ], p, p))

  if (dist_type == "shd") {

    # Hamming distance for all pairs at once: |a| + |b| - 2 a.b
    B = (U != 0) * 1
    s = rowSums(B)
    D = outer(s, s, "+") - 2 * tcrossprod(B)
    total_distance = as.vector(D %*% weights)

  } else if (dist_type == "sid") {

    S = sid_matrix(U, p)                     # S[g, h] = SID(true = G_g, est = G_h)
    total_distance = as.vector(crossprod(S, weights))

  } else {

    if (is.null(dist_fun)) {
      stop("dist_type = 'custom' requires a user-supplied dist_fun(A, B).")
    }
    graphs = lapply(seq_len(v), function(k) matrix(U[k, ], p, p))
    D = matrix(0, v, v)
    for (i in seq_len(v - 1L)) {
      Ai = graphs[[i]]
      for (j in (i + 1L):v) {
        D[i, j] = D[j, i] = dist_fun(Ai, graphs[[j]])
      }
    }
    total_distance = as.vector(D %*% weights)
  }

  best_index = which.min(total_distance)
  matrix(U[best_index, ], p, p)
}


# ---------------------------------------------------------------------------
# Internal helpers for SID
# ---------------------------------------------------------------------------

# Strict transitive closure of a 0/1 adjacency matrix (TRUE if a directed path
# of length >= 1 exists). Used both for reachability and the DAG check.
.transitive_closure = function(A) {
  p = nrow(A)
  P = A > 0
  if (p > 1L) {
    for (k in seq_len(ceiling(log2(p)))) {
      P = P | ((P %*% P) > 0)
    }
  }
  P
}

# Number of intervention effects i -> (all j) that are wrongly inferred when the
# true DAG is G and node i is adjusted for parent set Z (logical vector).
# G   : logical p x p adjacency of the true DAG (G[a, b] = a -> b)
# R   : reflexive reachability of G (R[a, b] = a is an ancestor of or equal to b)
# ch  : logical, children of i in G
# pa  : logical, parents of i in G
.sid_row = function(G, R, i, ch, pa, Z) {
  if (identical(Z, pa)) return(0L)          # parent adjustment is always valid

  p = length(Z)
  anyZ = any(Z)

  # Z claims j is a parent of i (so no effect), but G has i ~> j
  wrong_zero = Z & R[i, ]

  # "Bad" children of i have a descendant in Z; any j they reach has a
  # forbidden node in the adjustment set
  if (anyZ) {
    hasDescInZ = rowSums(R[, Z, drop = FALSE]) > 0
    bad = ch & hasDescInZ
  } else {
    hasDescInZ = rep(FALSE, p)
    bad = rep(FALSE, p)
  }
  forbidden = if (any(bad)) colSums(R[bad, , drop = FALSE]) > 0 else rep(FALSE, p)

  # d-connection from i given Z in G with edges i -> c removed for children c
  # that are not "bad" (these can only start causal or blocked paths).
  Gs = G
  Gs[i, ch & !bad] = FALSE

  # Bayes-ball (Koller & Friedman, Alg. 3.1), frontier-vectorised.
  # up   = reached from a child  (trail travelling upward)
  # down = reached from a parent (trail travelling downward)
  up_vis = down_vis = rep(FALSE, p)
  up_new = rep(FALSE, p); up_new[i] = TRUE
  down_new = rep(FALSE, p)
  repeat {
    up_vis = up_vis | up_new
    down_vis = down_vis | down_new
    go_parents  = (up_new & !Z) | (down_new & hasDescInZ)
    go_children = (up_new | down_new) & !Z
    nu = if (any(go_parents))  as.vector(Gs %*% go_parents)  > 0 else rep(FALSE, p)
    nd = if (any(go_children)) as.vector(go_children %*% Gs) > 0 else rep(FALSE, p)
    up_new = nu & !up_vis
    down_new = nd & !down_vis
    if (!any(up_new) && !any(down_new)) break
  }
  connected = (up_vis | down_vis) & !Z

  wrong = wrong_zero | (!Z & (forbidden | connected))
  wrong[i] = FALSE
  sum(wrong)
}

# Full (asymmetric) SID matrix between the unique graphs stored as rows of U.
# The work for node i only depends on (true graph, i, parent set of i in the
# estimate), so each distinct parent set is evaluated once per true graph and
# then reused for every estimate that shares it.
sid_matrix = function(U, p) {
  v = nrow(U)
  graphs = lapply(seq_len(v), function(k) {
    A = matrix(U[k, ], p, p) != 0
    diag(A) = FALSE
    A
  })

  closures = lapply(graphs, .transitive_closure)
  if (any(vapply(closures, function(P) any(diag(P)), logical(1L)))) {
    stop("SID distance requires all graphs to be DAGs.")
  }

  # For each node i: the distinct parent sets across all graphs, and which one
  # each graph uses
  pa_sets = vector("list", p)
  pa_id = matrix(0L, v, p)
  for (i in seq_len(p)) {
    cols = vapply(graphs, function(A) A[, i], logical(p))   # p x v
    if (p == 1L) cols = matrix(cols, 1L)
    key = apply(cols, 2L, function(z) paste(which(z), collapse = ","))
    uk = unique(key)
    pa_id[, i] = match(key, uk)
    pa_sets[[i]] = cols[, match(uk, key), drop = FALSE]
  }

  S = matrix(0, v, v)
  for (g in seq_len(v)) {
    G = graphs[[g]]
    R = closures[[g]]
    diag(R) = TRUE
    row_total = numeric(v)
    for (i in seq_len(p)) {
      sets = pa_sets[[i]]
      ch = G[i, ]
      pa = G[, i]
      cnt = vapply(seq_len(ncol(sets)),
                   function(k) .sid_row(G, R, i, ch, pa, sets[, k]),
                   integer(1L))
      row_total = row_total + cnt[pa_id[, i]]
    }
    S[g, ] = row_total
  }
  return(S)
}
