#' Generate synthetic DAG example data
#'
#' @description
#' Simulates data from a randomly generated sparse directed acyclic graph (DAG).
#' A random topological order is drawn, edges are placed only from earlier to
#' later nodes, and weights are given magnitudes drawn from \code{mag_range}
#' with independently drawn signs. Errors come from a finite normal mixture, so
#' the model is identified without a Gaussian assumption.
#'
#' This is not an estimation or causal discovery method. It exists to produce
#' example data sets for demonstrations, tests and simulation studies.
#'
#' @details
#' Because all edges respect a topological order, \eqn{B} is nilpotent: by
#' construction \eqn{\det(I - B) = 1} and \eqn{\rho(B) = 0}, so
#' \eqn{(I - B)^{-1}} always exists and no rejection step is needed.
#'
#' @param num_covariates Integer. Number of variables (nodes), \eqn{p}.
#' @param N Integer. Sample size.
#' @param M_input Integer. Number of components in the error mixture.
#' @param prob_sparsity Numeric in (0, 1). Probability that an ordered pair has
#'   NO edge, so the target edge density is \code{1 - prob_sparsity}. At most
#'   half of the ordered pairs can be edges in a DAG.
#' @param seed_input Integer. Random seed, for reproducibility.
#' @param mag_range Length-2 numeric. Range of absolute edge weights.
#' @param prob_positive Numeric in \eqn{[0,1]}. Probability an edge weight is positive.
#' @param max_parents Integer or NULL. Optional cap on the number of parents
#'   per node.
#' @param error_dist Error distribution: \code{"mixture"} for the normal
#'   mixture, \code{"laplace"}, or \code{"t"}. Component labels are drawn in
#'   every case and returned in \code{Z_matrix_true}; for \code{"laplace"} and
#'   \code{"t"} the errors themselves do not depend on them.
#' @param laplace_location,laplace_scale Location and scale of the Laplace
#'   errors, used when \code{error_dist = "laplace"}.
#' @param t_df Degrees of freedom, used when \code{error_dist = "t"}.
#' @param seed_structure,seed_weights Optional separate seeds for the graph
#'   structure and its weights. Each is applied locally and the previous random
#'   state is restored afterwards, so the same graph can be reused across
#'   replicates while the errors keep advancing from \code{seed_input}.
#'
#' @return A list containing
#' \describe{
#'   \item{data_matrix}{\eqn{N \times p} data matrix.}
#'   \item{Adjacency_matrix_true}{\eqn{p \times p} adjacency matrix, entry
#'     \eqn{(i,j)} nonzero when \eqn{j} is a parent of \eqn{i}.}
#'   \item{Causal_effect_matrix_true}{\eqn{p \times p} coefficient matrix \eqn{B}.}
#'   \item{Z_matrix_true}{\eqn{(pN) \times M} indicator matrix of true mixture
#'     memberships, in the layout used by \code{init_Z}.}
#'   \item{order}{The topological order used, earliest node first.}
#' }
#'
#' @seealso \code{\link{generates_examples_DCG}} for the cyclic version.
#' @export
#' @examples
#' ex <- generates_examples_DAG(num_covariates = 7, N = 250, M_input = 2,
#'                              prob_sparsity = 0.9, seed_input = 21)
#' dim(ex$data_matrix)
#' sum(ex$Adjacency_matrix_true)
generates_examples_DAG <- function(num_covariates, N, M_input, prob_sparsity,
                                   seed_input,
                                   mag_range     = c(0.4, 0.9),
                                   prob_positive = 0.5,
                                   max_parents   = NULL,
                                   error_dist    = c("mixture", "laplace", "t"),
                                   laplace_location = 0,
                                   laplace_scale    = 1/4,
                                   t_df             = 7,
                                   seed_structure   = NULL,
                                   seed_weights     = NULL) {

  error_dist <- match.arg(error_dist)

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

  ## Apply a local seed and restore the previous state afterwards, so a fixed
  ## graph can be paired with error streams that differ by replicate.
  .with_seed <- function(seed, expr) {
    if (is.null(seed)) return(expr())
    old <- if (exists(".Random.seed", envir = .GlobalEnv))
      get(".Random.seed", envir = .GlobalEnv) else NULL
    set.seed(seed)
    on.exit(if (!is.null(old)) assign(".Random.seed", old, envir = .GlobalEnv),
            add = TRUE)
    expr()
  }

  ## ------------------------------------------------------------------
  ## 1. STRUCTURE
  ## ------------------------------------------------------------------
  struct <- .with_seed(seed_structure, function() {
  ## random topological order: ord[k] is the node in position k
  ord <- sample(p)
  pos <- integer(p); pos[ord] <- seq_len(p)

  ## legal edges: u -> v whenever u precedes v
  g    <- expand.grid(u = seq_len(p), v = seq_len(p))
  pool <- g[pos[g$u] < pos[g$v], , drop = FALSE]
  pool_size <- nrow(pool)                       # = p*(p-1)/2

  total_cells <- p * (p - 1)
  target <- as.integer(round((1 - prob_sparsity) * total_cells))
  if (target > pool_size) {
    warning("Requested ", target, " edges but only ", pool_size,
            " are legal in a DAG on ", p, " nodes (max density ",
            sprintf("%.3f", pool_size / total_cells), "). Using all of them.",
            call. = FALSE)
    target <- pool_size
  }

  E     <- matrix(0, p, p)
  edges <- list()
  perm  <- sample.int(pool_size)                # random order to fill in
  n_par <- integer(p)
  taken <- 0L
  for (r in perm) {
    if (taken >= target) break
    u <- pool$u[r]; v <- pool$v[r]
    if (!is.null(max_parents) && n_par[v] >= max_parents) next
    E[v, u] <- 1                                # u -> v
    n_par[v] <- n_par[v] + 1L
    edges[[length(edges) + 1L]] <- c(v, u)
    taken <- taken + 1L
  }
  if (taken < target)
    warning("Only ", taken, " of ", target,
            " edges could be placed under max_parents = ", max_parents, ".",
            call. = FALSE)
  list(E = E, edges = edges, ord = ord)
  })
  E <- struct$E; edges <- struct$edges; ord <- struct$ord

  ## ------------------------------------------------------------------
  ## 2. WEIGHTS: magnitude first, then an independent sign
  ## ------------------------------------------------------------------
  B <- .with_seed(seed_weights, function() {
    n_edge    <- length(edges)
    magnitude <- stats::runif(n_edge, mag_range[1], mag_range[2])
    sign_draw <- stats::rbinom(n_edge, size = 1, prob = prob_positive)
    b         <- ifelse(sign_draw == 1, 1, -1) * magnitude

    Bw <- matrix(0, p, p)
    for (k in seq_along(edges)) {
      e <- edges[[k]]
      Bw[e[1], e[2]] <- b[k]
    }
    Bw
  })

  Adjacency_matrix_true <- (B != 0) * 1

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
      epsilon_true[z, i] <- switch(
        error_dist,
        mixture = rnorm(1, mu_epsilon[kk], sigma_epsilon[kk]),
        ## same closed form (and same single runif draw) as VGAM::rlaplace
        laplace = {
          u <- stats::runif(1)
          laplace_location - sign(u - 0.5) * laplace_scale *
            (log(2) + if (u < 0.5) log(u) else log1p(-u))
        },
        t = stats::rt(1, t_df)
      )
    }
  }

  A_inv <- solve(diag(p) - B)
  data_matrix <- matrix(0, N, p)
  for (i in seq_len(N)) data_matrix[i, ] <- (A_inv %*% epsilon_true[i, ])[, 1]

  list(data_matrix               = data_matrix,
       Adjacency_matrix_true     = Adjacency_matrix_true,
       Causal_effect_matrix_true = B,
       Z_matrix_true             = Z_matrix_true,
       order                     = ord)
}
