#' Generate synthetic DCG example data (helper function)
#'
#' @description
#' Helper function for generating reproducible synthetic data from a sparse
#' directed cyclic graph (DCG). It is primarily provided for the package
#' README, function examples, tests, and simulation studies.
#'
#' This function is not a causal discovery or graph-estimation procedure.
#' It generates simulated data with known graph structures and causal
#' coefficients so that users can demonstrate and evaluate the Bayesian
#' causal discovery methods implemented in the package.
#' To estimate cyclic causal structures from observed data, use
#' \code{\link{BayesDCG}}.
#'
#' @details
#' The function constructs a sparse DCG consisting of vertex-disjoint
#' directed cycles and acyclic edges between cycle blocks.
#' Nonzero causal-effect coefficients are generated with magnitudes
#' drawn from \code{mag_range} and independently assigned positive
#' or negative signs. Structural errors are generated from a finite
#' Gaussian mixture distribution.
#'
#' The generated adjacency matrix and causal-effect matrix are returned
#' as known ground truth, allowing users to compare the estimated
#' results from \code{\link{BayesDCG}} with the underlying simulated
#' causal structure.
#'
#' Because every cycle is vertex-disjoint, the determinant and spectral
#' radius have closed forms. Let \eqn{w_m} denote the product of the
#' coefficients around cycle \eqn{m} of length \eqn{L_m}:
#'
#' \deqn{\det(I-B)=\prod_m(1-w_m), \qquad
#'       \rho(B)=\max_m |w_m|^{1/L_m}.}
#'
#' Weights are redrawn until \eqn{\min_m |1-w_m| \ge}
#' \code{tol_sing} and \eqn{\rho(B) <} \code{rho_max}.
#' These restrictions ensure that the generated coefficient matrix
#' satisfies the specified stability and nonsingularity conditions.
#'
#' @param num_covariates Integer. Number of variables (nodes), \eqn{p}.
#' @param N Integer. Sample size.
#' @param M_input Integer. Number of components in the error mixture.
#' @param prob_sparsity Numeric in (0,1). Probability that an ordered
#'   pair has no edge, so the target edge density is
#'   \code{1 - prob_sparsity}.
#' @param seed_input Integer. Random seed for reproducibility.
#' @param n_cycles Integer. Number of vertex-disjoint cycles.
#'   The default, NULL, selects a value that fits within
#'   \eqn{p} vertices given \code{len_range}.
#' @param len_range Length-2 numeric. Range of cycle lengths.
#' @param mag_range Length-2 numeric. Range of absolute edge weights.
#' @param prob_positive Numeric in \eqn{[0,1]}. Probability that
#'   an edge weight is positive.
#' @param tol_sing Numeric. Required margin on
#'   \eqn{\min_m |1-w_m|}.
#' @param rho_max Numeric. Upper bound on the spectral radius.
#' @param max_tries Integer. Maximum number of weight redraws.
#'
#' @return
#' A list containing simulated observations and the corresponding
#' true model parameters:
#' \describe{
#'   \item{data_matrix}{
#'     \eqn{N \times p} simulated data matrix.
#'   }
#'   \item{Adjacency_matrix_true}{
#'     \eqn{p \times p} true adjacency matrix, where entry
#'     \eqn{(i,j)} is nonzero when \eqn{j} is a parent of \eqn{i}.
#'   }
#'   \item{Causal_effect_matrix_true}{
#'     \eqn{p \times p} true causal-effect coefficient matrix
#'     \eqn{B}.
#'   }
#'   \item{Z_matrix_true}{
#'     \eqn{(pN) \times M} indicator matrix of true mixture
#'     memberships, in the layout used by \code{init_Z}.
#'   }
#'   \item{cycles}{
#'     List of node sets forming the generated directed cycles.
#'   }
#'   \item{rho}{
#'     Spectral radius of the generated causal-effect matrix.
#'   }
#' }
#'
#' @seealso
#' \code{\link{generates_examples_DAG}} for generating synthetic
#' acyclic data, and \code{\link{BayesDCG}} for Bayesian causal
#' discovery in cyclic models.
#'
#' @export
#'
#' @examples
#' # Generate synthetic DCG data for demonstration
#' ex <- generates_examples_DCG(
#'   num_covariates = 7,
#'   N = 250,
#'   M_input = 2,
#'   prob_sparsity = 0.9,
#'   seed_input = 21
#' )
#'
#' # Inspect the generated data and true graph
#' dim(ex$data_matrix)
#' sum(ex$Adjacency_matrix_true)
#'
#' # Inspect the spectral radius
#' ex$rho
generates_examples_DCG <- function(num_covariates, N, M_input, prob_sparsity,
                                   seed_input,
                                   n_cycles      = NULL,
                                   len_range     = c(2, 4),
                                   mag_range     = c(0.4, 0.9),
                                   prob_positive = 0.5,
                                   tol_sing      = 0.05,
                                   rho_max       = 0.95,
                                   max_tries     = 1000) {

  p <- as.integer(num_covariates)
  if (p < 2L)       stop("num_covariates must be at least 2.", call. = FALSE)
  if (N < 1L)       stop("N must be at least 1.", call. = FALSE)
  if (M_input < 1L) stop("M_input must be at least 1.", call. = FALSE)
  if (prob_sparsity <= 0 || prob_sparsity >= 1)
    stop("prob_sparsity must be strictly between 0 and 1.", call. = FALSE)
  if (length(mag_range) != 2 || mag_range[1] < 0 || mag_range[1] > mag_range[2])
    stop("mag_range must be c(lower, upper) with 0 <= lower <= upper.", call. = FALSE)
  if (prob_positive < 0 || prob_positive > 1)
    stop("prob_positive must be between 0 and 1.", call. = FALSE)

  set.seed(seed_input)

  ## Cycles are vertex-disjoint, so no cycle may be longer than p and
  ## n_cycles * (longest cycle) must fit inside p.
  len_range <- c(min(len_range[1], p), min(len_range[2], p))
  if (is.null(n_cycles))
    n_cycles <- max(1L, min(max(2L, floor(p / 10)), floor(p / len_range[2])))

  ## ------------------------------------------------------------------
  ## 1. STRUCTURE
  ## ------------------------------------------------------------------
  cycle_lengths <- sample(seq(len_range[1], len_range[2]), n_cycles,
                          replace = TRUE)
  if (sum(cycle_lengths) > p)
    stop("sum(cycle_lengths) = ", sum(cycle_lengths), " exceeds p = ", p,
         ". Reduce n_cycles or len_range.", call. = FALSE)

  ## disjoint node sets
  perm   <- sample(p)
  cycles <- vector("list", length(cycle_lengths))
  pos    <- 0L
  for (m in seq_along(cycle_lengths)) {
    L <- cycle_lengths[m]
    cycles[[m]] <- perm[(pos + 1L):(pos + L)]
    pos <- pos + L
  }
  leftover <- if (pos < p) perm[(pos + 1L):p] else integer(0)

  ## block order: cycles first, then singletons
  blocks <- c(cycles, lapply(leftover, function(v) v))
  blk    <- integer(p)
  for (m in seq_along(blocks)) blk[blocks[[m]]] <- m

  ## cycle edges: cyc[k] -> cyc[k+1]
  E <- matrix(0, p, p)
  for (cyc in cycles) {
    L <- length(cyc)
    for (k in seq_len(L)) E[cyc[k %% L + 1L], cyc[k]] <- 1
  }
  n_cycle_edges <- sum(cycle_lengths)

  ## Pool of legal acyclic edges: block(u) < block(v). Anything inside a
  ## cycle block would break the in/out-degree = 1 condition, and anything
  ## backwards would create a new cycle or merge two components.
  g    <- expand.grid(u = seq_len(p), v = seq_len(p))
  pool <- g[blk[g$u] < blk[g$v], , drop = FALSE]
  pool_size <- nrow(pool)

  total_cells <- p * (p - 1)
  target  <- as.integer(round((1 - prob_sparsity) * total_cells))
  n_extra <- target - n_cycle_edges
  if (n_extra < 0) {
    warning("Target of ", target, " edges is below the ", n_cycle_edges,
            " cycle edges required. Using cycle edges only (density ",
            sprintf("%.4f", n_cycle_edges / total_cells),
            "). Reduce n_cycles or len_range for a sparser graph.", call. = FALSE)
    n_extra <- 0L
  }
  if (n_extra > pool_size) {
    warning("Requested ", n_extra, " acyclic edges but only ", pool_size,
            " are legal (max density ",
            sprintf("%.3f", (pool_size + n_cycle_edges) / total_cells),
            "). Using all of them.", call. = FALSE)
    n_extra <- pool_size
  }

  ## sampled without replacement so the edge count is exact
  dag_edges <- list()
  if (n_extra > 0) {
    pick <- sample.int(pool_size, n_extra)
    for (r in pick) {
      u <- pool$u[r]; v <- pool$v[r]
      E[v, u] <- 1                                   # u -> v
      dag_edges[[length(dag_edges) + 1L]] <- c(v, u)
    }
  }

  ## ------------------------------------------------------------------
  ## 2. WEIGHTS: magnitude first, then an independent sign
  ## ------------------------------------------------------------------
  draw_w <- function(n) {
    magnitude <- stats::runif(n, mag_range[1], mag_range[2])
    sign_draw <- stats::rbinom(n, size = 1, prob = prob_positive)
    ifelse(sign_draw == 1, 1, -1) * magnitude
  }

  k <- length(cycles)
  B <- NULL
  for (attempt in seq_len(max_tries)) {
    Btry <- matrix(0, p, p)

    ## cycle edges: the only ones that affect det(I - B) and rho(B)
    w <- numeric(k)
    for (m in seq_len(k)) {
      cyc <- cycles[[m]]
      L   <- length(cyc)
      b   <- draw_w(L)
      for (i in seq_len(L)) Btry[cyc[i %% L + 1L], cyc[i]] <- b[i]
      w[m] <- prod(b)
    }

    ## acyclic edges
    if (length(dag_edges)) {
      bd <- draw_w(length(dag_edges))
      for (j in seq_along(dag_edges)) {
        e <- dag_edges[[j]]
        Btry[e[1], e[2]] <- bd[j]
      }
    }

    if (k == 0L) { B <- Btry; break }   # no cycles: I - B unit triangular

    sing_margin <- min(abs(1 - w))
    rho         <- max(abs(w)^(1 / sapply(cycles, length)))
    if (sing_margin >= tol_sing && rho < rho_max) { B <- Btry; break }
  }
  if (is.null(B))
    stop("Could not achieve min|1 - w_m| >= ", tol_sing, " and rho < ", rho_max,
         " within ", max_tries, " draws. Try reducing the upper end of ",
         "mag_range.", call. = FALSE)

  Adjacency_matrix_true <- (B != 0) * 1
  rho_final <- if (k == 0L) 0 else max(abs(w)^(1 / sapply(cycles, length)))

  ## ------------------------------------------------------------------
  ## 3. ERRORS and DATA
  ## ------------------------------------------------------------------
  M <- as.integer(M_input)
  mu_epsilon    <- if (M == 1L) 0   else seq(-0.5, 0.5, length.out = M)
  sigma_epsilon <- if (M == 1L) 0.1 else seq( 0.1, 0.3, length.out = M)

  epsilon_true  <- matrix(0, N, p)
  Z_matrix_true <- matrix(0, p * N, M)
  for (i in seq_len(p)) {
    for (z in seq_len(N)) {
      kk <- sample(seq_len(M), size = 1, replace = TRUE)
      Z_matrix_true[(i - 1L) * N + z, kk] <- 1
      epsilon_true[z, i] <- rnorm(1, mu_epsilon[kk], sigma_epsilon[kk])
    }
  }

  data_matrix <- t(solve(diag(p) - B, t(epsilon_true)))

  list(data_matrix               = data_matrix,
       Adjacency_matrix_true     = Adjacency_matrix_true,
       Causal_effect_matrix_true = B,
       Z_matrix_true             = Z_matrix_true,
       cycles                    = cycles,
       rho                       = rho_final)
}
