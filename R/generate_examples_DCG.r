#' Generate synthetic DCG example data
#'
#' @description
#' Simulates data from a randomly generated sparse directed cyclic graph (DCG).
#' The graph is built as a set of vertex-disjoint cycles plus acyclic edges
#' between cycle blocks, then given weights whose magnitudes are drawn from
#' \code{mag_range} with independently drawn signs. Errors come from a finite
#' normal mixture, so the model is identified without a Gaussian assumption.
#'
#' This is not an estimation or causal discovery method. It exists to produce
#' example data sets for demonstrations, tests and simulation studies.
#'
#' @details
#' Because every cycle is vertex-disjoint, the determinant and spectral radius
#' have closed forms: with \eqn{w_m} the product of the weights around cycle
#' \eqn{m} of length \eqn{L_m},
#' \deqn{\det(I - B) = \prod_m (1 - w_m), \qquad \rho(B) = \max_m |w_m|^{1/L_m}.}
#' Weights are redrawn until \eqn{\min_m |1 - w_m| \ge} \code{tol_sing} and
#' \eqn{\rho(B) <} \code{rho_max}, so \eqn{(I - B)^{-1}} is stable and well
#' conditioned.
#'
#' @param num_covariates Integer. Number of variables (nodes), \eqn{p}.
#' @param N Integer. Sample size.
#' @param M_input Integer. Number of components in the error mixture.
#' @param prob_sparsity Numeric in (0, 1). Probability that an ordered pair has
#'   NO edge, so the target edge density is \code{1 - prob_sparsity}.
#' @param seed_input Integer. Random seed, for reproducibility.
#' @param n_cycles Integer. Number of vertex-disjoint cycles. The default, NULL,
#'   picks a value that fits in \eqn{p} vertices given \code{len_range}.
#' @param len_range Length-2 numeric. Range of cycle lengths.
#' @param mag_range Length-2 numeric. Range of absolute edge weights.
#' @param prob_positive Numeric in \eqn{[0,1]}. Probability an edge weight is positive.
#' @param tol_sing Numeric. Required margin on \eqn{\min_m |1 - w_m|}.
#' @param rho_max Numeric. Upper bound required on the spectral radius.
#' @param max_tries Integer. Maximum weight redraws before giving up.
#'
#' @return A list containing
#' \describe{
#'   \item{data_matrix}{\eqn{N \times p} data matrix.}
#'   \item{Adjacency_matrix_true}{\eqn{p \times p} adjacency matrix, entry
#'     \eqn{(i,j)} nonzero when \eqn{j} is a parent of \eqn{i}.}
#'   \item{Causal_effect_matrix_true}{\eqn{p \times p} coefficient matrix \eqn{B}.}
#'   \item{Z_matrix_true}{\eqn{(pN) \times M} indicator matrix of true mixture
#'     memberships, in the layout used by \code{init_Z}.}
#'   \item{cycles}{List of the node sets forming each cycle.}
#'   \item{rho}{Spectral radius of \eqn{B}.}
#' }
#'
#' @seealso \code{\link{generates_examples_DAG}} for the acyclic version.
#' @export
#' @examples
#' ex <- generates_examples_DCG(num_covariates = 7, N = 250, M_input = 2,
#'                              prob_sparsity = 0.9, seed_input = 21)
#' dim(ex$data_matrix)
#' sum(ex$Adjacency_matrix_true)
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
