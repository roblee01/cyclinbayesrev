#' Initialize Bayesian Causal Discovery Samplers (Helper Function)
#'
#' @description
#' Helper function for constructing an initial parameter state for
#' \code{\link{BayesDAG}} or \code{\link{BayesDCG}}.
#' It is primarily provided to support the reproducible examples
#' in the package README, function demonstrations, and simulation studies.
#'
#' This function is not a standalone causal discovery or estimation
#' procedure. Instead, it prepares the initial parameter values
#' required to run the Bayesian causal discovery samplers.
#' Users can supply a preliminary coefficient matrix, such as one
#' obtained from \code{\link{directlingam_seed}}, or initialize
#' from an empty graph.
#'
#' @details
#' The function constructs a consistent initial state for the
#' adjacency matrix, causal-effect coefficients, error-mixture
#' parameters, mixture weights, and hyperparameters.
#'
#' Given a seed coefficient matrix, the function first determines
#' the initial graph structure and computes the corresponding
#' residuals. It then initializes the Gaussian mixture parameters
#' using these residuals. This provides an initial state in which
#' the graph and error model parameters are mutually consistent.
#'
#' Seed coefficients smaller than \code{edge_threshold} in
#' absolute value are removed. The coefficient matrix is also
#' rescaled if its spectral radius reaches \code{rho_target},
#' ensuring that the initialized model satisfies the specified
#' stability condition.
#'
#' When \code{B_seed = NULL}, initialization begins from an
#' empty graph, without requiring an external estimation method.
#'
#' The resulting parameter values can be passed to
#' \code{\link{BayesDAG}} or \code{\link{BayesDCG}} as initial
#' states. These values serve only as starting points,
#' posterior inference is subsequently performed by the
#' corresponding Bayesian sampler.
#'
#' @param B_seed \eqn{p \times p} seed coefficient matrix,
#'   or NULL to initialize from an empty graph.
#' @param data_matrix \eqn{N \times p} numeric data matrix.
#' @param N Integer. Number of observations.
#' @param num_covariates Integer. Number of variables, \eqn{p}.
#' @param M Integer. Number of mixture components.
#' @param a_mu,b_mu,a_tao,b_tao,alpha Error-mixture
#'   hyperparameters, as passed to the sampler.
#' @param a_gamma,b_gamma,a_gamma_1,b_gamma_1 Graph and slab
#'   hyperparameters, as passed to the sampler.
#' @param edge_threshold Numeric. Seed coefficients smaller
#'   than this value in absolute magnitude are treated as absent.
#' @param rho_target Numeric. Coefficients are rescaled if
#'   the spectral radius reaches this value.
#' @param settle_sweeps Integer. Number of Gaussian-mixture
#'   Gibbs sweeps performed with the seed graph fixed
#'   before initialization is completed.
#' @param verbose Logical. Whether to print a summary of
#'   the initialized state.
#' @param acyclic Logical. If TRUE, edges are removed in
#'   increasing order of absolute coefficient magnitude
#'   until the graph is acyclic, as required by
#'   \code{\link{BayesDAG}}.
#'
#' @return
#' A list containing the initialized model parameters:
#' \describe{
#'   \item{Adjacency_matrix}{
#'     Initial adjacency matrix.
#'   }
#'   \item{Causal_effect_matrix}{
#'     Initial causal-effect coefficient matrix.
#'   }
#'   \item{Z_matrix}{
#'     Initial mixture-component allocation indicators.
#'   }
#'   \item{mu_mat}{
#'     Initial mixture-component means.
#'   }
#'   \item{tao_mat}{
#'     Initial mixture-component variances.
#'   }
#'   \item{pi_mat}{
#'     Initial mixture weights.
#'   }
#'   \item{gamma_1}{
#'     Initial slab variance.
#'   }
#'   \item{gamma_result}{
#'     Initial edge-inclusion probability.
#'   }
#'   \item{log_post}{
#'     Initial log-posterior value.
#'   }
#' }
#'
#' The returned values can be passed to a Bayesian sampler
#' using \code{\link{bayes_init_args}}.
#'
#' @seealso
#' \code{\link{directlingam_seed}} for obtaining a preliminary
#' coefficient matrix,
#' \code{\link{bayes_init_args}} for passing initialized
#' parameters to a sampler,
#' \code{\link{BayesDAG}} and \code{\link{BayesDCG}} for
#' Bayesian causal discovery.
#'
#' @export
#'
#' @examples
#' # Generate synthetic DAG data
#' ex <- generates_examples_DAG(
#'   num_covariates = 7,
#'   N = 150,
#'   M_input = 2,
#'   prob_sparsity = 0.9,
#'   seed_input = 21
#' )
#'
#' # Construct an initial state from an empty graph
#' init_state <- init_from_seed(
#'   B_seed = NULL,
#'   data_matrix = ex$data_matrix,
#'   N = 150,
#'   num_covariates = 7,
#'   M = 2,
#'   a_mu = 0,
#'   b_mu = 2,
#'   a_tao = 2,
#'   b_tao = 1,
#'   alpha = 1,
#'   a_gamma = 0.5,
#'   b_gamma = 0.5,
#'   a_gamma_1 = 2,
#'   b_gamma_1 = 1,
#'   acyclic = TRUE,
#'   verbose = FALSE
#' )
#'
#' # Fit BayesDAG using the initialized parameters
#' fit <- do.call(
#'   BayesDAG,
#'   c(
#'     list(
#'       data_matrix = ex$data_matrix,
#'       M = 2,
#'       num_iter = 2000
#'     ),
#'     bayes_init_args(init_state)
#'   )
#' )
init_from_seed <- function(
    B_seed,
    data_matrix,
    N, num_covariates, M,
    a_mu, b_mu, a_tao, b_tao, alpha,
    a_gamma, b_gamma, a_gamma_1, b_gamma_1,
    edge_threshold  = 0.05,
    rho_target      = 0.95,
    settle_sweeps   = 30,
    verbose         = TRUE,
    acyclic         = FALSE
) {
  p  <- num_covariates
  Ip <- diag(p)

  if (nrow(data_matrix) != N || ncol(data_matrix) != p)
    stop("data_matrix must be N x num_covariates.", call. = FALSE)

  ## --- 1. graph + coefficients from the seed --------------------------------
  B <- if (is.null(B_seed)) matrix(0, p, p) else as.matrix(B_seed)
  if (nrow(B) != p || ncol(B) != p)
    stop("B_seed must be num_covariates x num_covariates.", call. = FALSE)
  B[abs(B) < edge_threshold] <- 0     # prune noise-level seed edges
  diag(B) <- 0

  ## stability: rescale so spectral radius < rho_target
  rho <- max(Mod(eigen(B, only.values = TRUE)$values))
  if (rho >= rho_target && rho > 0) B <- B * (rho_target / rho)

  ## optionally force a DAG, dropping the weakest edge in turn
  if (acyclic) {
    while (!.init_is_acyclic(B)) {
      nz <- which(B != 0)
      B[nz[which.min(abs(B[nz]))]] <- 0
    }
  }

  Adj <- (B != 0) * 1

  ## --- 2. residuals implied by the seed graph -------------------------------
  eps <- t((Ip - B) %*% t(data_matrix))     # N x p  : eps[z, i]

  ## --- 3. mixture fit to residuals, per node --------------------------------
  ## mu at residual quantiles, tau at the marginal variance, Z by nearest
  ## component, then let the real Gibbs updates settle it.
  mu_mat  <- matrix(0, p, M)
  tao_mat <- matrix(0, p, M)
  Z_mat   <- matrix(0, p * N, M)
  probs   <- seq(0.5 / M, 1 - 0.5 / M, length.out = M)

  for (i in 1:p) {
    ri <- eps[, i]
    mu_mat[i, ]  <- as.numeric(stats::quantile(ri, probs))
    tao_mat[i, ] <- max(stats::var(ri), 1e-4)
    d  <- outer(ri, mu_mat[i, ], function(x, m) (x - m)^2)
    zi <- max.col(-d, ties.method = "first")
    rows <- ((i - 1) * N + 1):(i * N)
    Z_mat[cbind(rows, zi)] <- 1
  }

  pi_mat <- matrix(1 / M, p, M)

  ## settle the mixture on the FIXED seed graph (coefficients held at B)
  for (s in seq_len(settle_sweeps)) {
    mu_mat  <- mu_fun(Z_mat, a_mu, b_mu, tao_mat, eps, p, M, N)
    tao_mat <- tao_fun(Z_mat, a_tao, b_tao, mu_mat, eps, p, M, N)
    Z_mat   <- Z_matrix_fun(Z_mat, eps, mu_mat, tao_mat, pi_mat, p, N, M)
    pi_mat  <- pi_fun(Z_mat, p, alpha, M, N)
  }

  ## --- 4. scalars, seeded from the state ------------------------------------
  nz <- B[B != 0]
  gamma_1 <- if (length(nz)) mean(nz^2) else b_gamma_1 / (a_gamma_1 - 1)
  gamma_result <- (a_gamma + sum(Adj)) / (a_gamma + b_gamma + p * (p - 1))

  ## --- 5. coherence check: score the seeded state ---------------------------
  lp <- sum(Metropolis_hastings_portions_cpp(
    data_matrix, Adj, B, Z_mat, mu_mat, tao_mat,
    N, M, gamma_1, gamma_result))

  if (verbose)
    cat(sprintf(paste0("init_from_seed: edges %d  rho %.3f  gamma_1 %.3f  ",
                       "gamma_result %.3f  log-post %.1f\n"),
                sum(Adj), max(Mod(eigen(B, only.values = TRUE)$values)),
                gamma_1, gamma_result, lp))

  list(
    Adjacency_matrix     = Adj,
    Causal_effect_matrix = B,
    Z_matrix             = Z_mat,
    mu_mat               = mu_mat,
    tao_mat              = tao_mat,
    pi_mat               = pi_mat,
    gamma_1              = gamma_1,
    gamma_result         = gamma_result,
    log_post             = lp
  )
}


#' Convert a seeded state into sampler arguments
#'
#' @description
#' Renames the output of \code{\link{init_from_seed}} to the \code{init_*}
#' argument names used by \code{\link{BayesDCG}} and \code{\link{BayesDAG}},
#' so the state can be passed straight through with \code{do.call}.
#'
#' @param state A list as returned by \code{\link{init_from_seed}} or
#'   \code{\link{init_multistart_from_seed}}.
#' @return A named list of \code{init_*} arguments.
#' @seealso \code{\link{init_from_seed}}
#' @export
#' @examples
#' # do.call(BayesDCG, c(list(data_matrix = X, M = 5, num_iter = 1000,
#' #                          burn_in_iterations = 700),
#' #                     bayes_init_args(init_state)))
bayes_init_args <- function(state) {
  need <- c("Adjacency_matrix", "Causal_effect_matrix", "mu_mat", "tao_mat",
            "pi_mat", "Z_matrix", "gamma_1", "gamma_result")
  missing <- setdiff(need, names(state))
  if (length(missing))
    stop("state is missing: ", paste(missing, collapse = ", "), call. = FALSE)

  list(init_Adjacency     = state$Adjacency_matrix,
       init_Causal_effect = state$Causal_effect_matrix,
       init_mu            = state$mu_mat,
       init_tao           = state$tao_mat,
       init_pi            = state$pi_mat,
       init_Z             = state$Z_matrix,
       init_gamma_1       = state$gamma_1,
       init_gamma_result  = state$gamma_result)
}


#' Multi-start warm start from a perturbed seed
#'
#' @description
#' Perturbs the seed matrix \code{K} ways, builds a state from each with
#' \code{\link{init_from_seed}}, and returns the one with the highest
#' log-posterior. This restores some of the initialisation diversity that a
#' single deterministic seed removes, while keeping its structure.
#'
#' @param B_seed \eqn{p \times p} seed coefficient matrix.
#' @param K Integer. Number of starts; the first is the unperturbed seed.
#' @param perturb_sd Numeric. Standard deviation of the jitter added to
#'   existing edges.
#' @param data_matrix,N,num_covariates,M Data and dimensions, as in
#'   \code{\link{init_from_seed}}.
#' @param a_mu,b_mu,a_tao,b_tao,alpha,a_gamma,b_gamma,a_gamma_1,b_gamma_1
#'   Hyperparameters, as in \code{\link{init_from_seed}}.
#' @param edge_threshold,rho_target,settle_sweeps,acyclic Passed to
#'   \code{\link{init_from_seed}}.
#' @param verbose Logical. Print the score of each start.
#'
#' @return The best state, as returned by \code{\link{init_from_seed}}, with an
#'   extra \code{all_scores} element holding every start's log-posterior.
#' @seealso \code{\link{init_from_seed}}
#' @export
init_multistart_from_seed <- function(
    B_seed, K = 5, perturb_sd = 0.1,
    data_matrix, N, num_covariates, M,
    a_mu, b_mu, a_tao, b_tao, alpha,
    a_gamma, b_gamma, a_gamma_1, b_gamma_1,
    edge_threshold = 0.05, rho_target = 0.95, settle_sweeps = 30,
    acyclic = FALSE, verbose = TRUE
) {
  best <- NULL; best_lp <- -Inf; scores <- numeric(K)

  for (k in seq_len(K)) {
    Bk <- B_seed
    nz <- Bk != 0
    if (k > 1) Bk[nz] <- Bk[nz] + stats::rnorm(sum(nz), 0, perturb_sd)

    st <- init_from_seed(
      Bk, data_matrix, N, num_covariates, M,
      a_mu, b_mu, a_tao, b_tao, alpha,
      a_gamma, b_gamma, a_gamma_1, b_gamma_1,
      edge_threshold, rho_target, settle_sweeps,
      verbose = FALSE, acyclic = acyclic)

    scores[k] <- st$log_post
    if (verbose)
      cat(sprintf("  start %d/%d: edges %d  log-post %.1f\n",
                  k, K, sum(st$Adjacency_matrix), st$log_post))
    if (st$log_post > best_lp) { best_lp <- st$log_post; best <- st }
  }

  if (verbose)
    cat(sprintf(">> selected seed: log-post %.1f  edges %d\n",
                best_lp, sum(best$Adjacency_matrix)))
  best$all_scores <- scores
  best
}


#' Is an adjacency pattern acyclic?
#'
#' Kahn's algorithm, used to prune a seed graph down to a DAG.
#'
#' @param A Square numeric matrix; nonzero entries are edges.
#' @return TRUE when the graph has no directed cycle.
#' @noRd
.init_is_acyclic <- function(A) {
  A <- (A != 0) * 1L
  diag(A) <- 0L
  p <- nrow(A)
  indeg <- colSums(A)
  queue <- which(indeg == 0L)
  removed <- 0L
  while (length(queue) > 0L) {
    v <- queue[1L]; queue <- queue[-1L]; removed <- removed + 1L
    out <- which(A[v, ] != 0L)
    if (length(out)) {
      indeg[out] <- indeg[out] - 1L
      A[v, out] <- 0L
      queue <- c(queue, out[indeg[out] == 0L])
    }
  }
  removed == p
}
