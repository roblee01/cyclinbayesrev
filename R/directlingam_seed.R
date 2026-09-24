#' Seed coefficient matrix from DirectLiNGAM
#'
#' @description
#' Estimates a coefficient matrix to use as a warm start for
#' \code{\link{BayesDCG}} or \code{\link{BayesDAG}}, via
#' \code{\link[pcalg]{lingam}} when \pkg{pcalg} is installed and a built-in
#' implementation otherwise.
#'
#' @details
#' The result is intended only as a starting point: it is passed to
#' \code{\link{init_from_seed}}, which prunes small coefficients, rescales for
#' stability and fits the error mixture around the implied residuals.
#'
#' The returned matrix follows the same convention as the samplers: entry
#' \eqn{(i, j)} is the coefficient of variable \eqn{j} in the equation for
#' variable \eqn{i}, so \eqn{x = Bx + e}.
#'
#' @param data_matrix \eqn{N \times p} numeric data matrix.
#' @param use_pcalg Logical. Use \code{\link[pcalg]{lingam}} when \pkg{pcalg}
#'   is available. Set FALSE to always use the built-in implementation.
#'
#' @return A \eqn{p \times p} coefficient matrix with a zero diagonal.
#' @seealso \code{\link{directlingam_R}}, \code{\link{init_from_seed}}
#' @export
#' @examples
#' ex <- generates_examples_DAG(num_covariates = 6, N = 200, M_input = 2,
#'                              prob_sparsity = 0.9, seed_input = 3)
#' B_seed <- directlingam_seed(ex$data_matrix)
#' dim(B_seed)
directlingam_seed <- function(data_matrix, use_pcalg = TRUE) {
  data_matrix <- as.matrix(data_matrix)
  p <- ncol(data_matrix)

  B <- NULL
  if (use_pcalg && requireNamespace("pcalg", quietly = TRUE)) {
    B <- tryCatch({
      fit <- pcalg::lingam(data_matrix)
      ## pcalg has returned a plain list in some versions and an S4 object in
      ## others; accept either.
      out <- if (is.list(fit) && !is.null(fit$Bpruned)) fit$Bpruned
             else if (isS4(fit) && methods::.hasSlot(fit, "Bpruned")) methods::slot(fit, "Bpruned")
             else NULL
      if (is.null(out)) NULL else as.matrix(out)
    }, error = function(e) {
      warning("pcalg::lingam failed (", conditionMessage(e),
              "); using the built-in DirectLiNGAM.", call. = FALSE)
      NULL
    })
  }

  if (is.null(B)) B <- directlingam_R(data_matrix)

  if (nrow(B) != p || ncol(B) != p)
    stop("Seed matrix is not ", p, " x ", p, ".", call. = FALSE)
  diag(B) <- 0
  B[!is.finite(B)] <- 0
  B
}


#' DirectLiNGAM in base R
#'
#' @description
#' Pure-R implementation of DirectLiNGAM (Shimizu et al., 2011), used by
#' \code{\link{directlingam_seed}} when \pkg{pcalg} is not installed. It needs
#' no other packages.
#'
#' @details
#' The causal order is found one variable at a time. At each step every
#' remaining variable is scored by how much residual dependence it leaves in
#' the others, using the pairwise likelihood ratio of Hyvarinen and Smith with
#' the standard entropy approximation; the least dependent variable is taken as
#' the most exogenous, the others are replaced by their residuals on it, and the
#' search repeats. Coefficients are then obtained by regressing each variable on
#' its predecessors in that order, so the result is acyclic by construction.
#'
#' @param data_matrix \eqn{N \times p} numeric data matrix.
#' @param prune_threshold Numeric. Coefficients smaller than this in absolute
#'   value are set to zero, on top of any significance pruning.
#' @param prune_alpha Numeric or NULL. Within each regression, predictors whose
#'   two-sided t-test p-value exceeds this are dropped and the equation is
#'   refitted. Regressing on every predecessor otherwise leaves small nonzero
#'   coefficients on absent edges. Set NULL to keep the plain least-squares
#'   fit.
#'
#' @return A \eqn{p \times p} coefficient matrix, entry \eqn{(i, j)} being the
#'   coefficient of \eqn{j} in the equation for \eqn{i}. The attribute
#'   \code{"causal_order"} holds the estimated order, most exogenous first.
#' @references Shimizu, S., Inazumi, T., Sogawa, Y., Hyvarinen, A., Kawahara,
#'   Y., Washio, T., Hoyer, P. O. and Bollen, K. (2011). DirectLiNGAM: a direct
#'   method for learning a linear non-Gaussian structural equation model.
#'   \emph{Journal of Machine Learning Research}, 12, 1225-1248.
#' @seealso \code{\link{directlingam_seed}}
#' @export
#' @examples
#' ex <- generates_examples_DAG(num_covariates = 6, N = 200, M_input = 2,
#'                              prob_sparsity = 0.9, seed_input = 3)
#' B <- directlingam_R(ex$data_matrix)
#' attr(B, "causal_order")
directlingam_R <- function(data_matrix, prune_threshold = 0,
                           prune_alpha = 0.05) {
  X <- as.matrix(data_matrix)
  if (!all(is.finite(X))) stop("data_matrix must be finite.", call. = FALSE)
  N <- nrow(X); p <- ncol(X)
  if (p < 2L) stop("data_matrix needs at least 2 columns.", call. = FALSE)

  ## --- search the causal order ---------------------------------------------
  W <- scale(X)                       # work on standardised copies
  remaining <- seq_len(p)
  causal_order <- integer(0)

  while (length(remaining) > 1L) {
    scores <- vapply(remaining, function(i) {
      total <- 0
      for (j in remaining) {
        if (j == i) next
        xi <- .dl_std(W[, i]); xj <- .dl_std(W[, j])
        rij <- .dl_std(.dl_residual(xi, xj))   # xi given xj
        rji <- .dl_std(.dl_residual(xj, xi))   # xj given xi
        ## positive when i is the more plausible cause of j
        diff_mi <- (.dl_entropy(xj) + .dl_entropy(rij)) -
                   (.dl_entropy(xi) + .dl_entropy(rji))
        total <- total + min(0, diff_mi)^2
      }
      total
    }, numeric(1))

    k <- remaining[which.min(scores)]
    causal_order <- c(causal_order, k)
    others <- setdiff(remaining, k)
    for (j in others) W[, j] <- .dl_residual(W[, j], W[, k])
    remaining <- others
  }
  causal_order <- c(causal_order, remaining)

  ## --- coefficients: regress each variable on its predecessors -------------
  B <- matrix(0, p, p)
  if (p >= 2L) {
    for (idx in 2:p) {
      i     <- causal_order[idx]
      preds <- causal_order[seq_len(idx - 1L)]
      keep <- preds
      if (!is.null(prune_alpha) && length(keep)) {
        ## drop predecessors that add nothing, then refit on the survivors
        fit <- stats::lm.fit(cbind(1, X[, keep, drop = FALSE]), X[, i])
        dfr <- N - length(keep) - 1L
        if (dfr > 0) {
          rss <- sum(fit$residuals^2)
          R <- qr.R(fit$qr)
          xtxinv <- tryCatch(chol2inv(R), error = function(e) NULL)
          if (!is.null(xtxinv)) {
            se <- sqrt(pmax(diag(xtxinv), 0) * rss / dfr)
            pv <- 2 * stats::pt(abs(fit$coefficients / se), dfr, lower.tail = FALSE)
            keep <- keep[is.finite(pv[-1L]) & pv[-1L] <= prune_alpha]
          }
        }
      }
      if (length(keep)) {
        design <- cbind(1, X[, keep, drop = FALSE])
        coef <- tryCatch(qr.solve(design, X[, i]), error = function(e) NULL)
        if (!is.null(coef)) B[i, keep] <- coef[-1L]   # drop the intercept
      }
    }
  }

  B[abs(B) < prune_threshold] <- 0
  B[!is.finite(B)] <- 0
  diag(B) <- 0
  attr(B, "causal_order") <- causal_order
  B
}


#' Standardise a vector
#' @param u Numeric vector.
#' @return u centred and scaled to unit variance; unchanged if it is constant.
#' @noRd
.dl_std <- function(u) {
  s <- stats::sd(u)
  if (!is.finite(s) || s == 0) return(u - mean(u))
  (u - mean(u)) / s
}

#' Residual of u after regressing on v
#' @param u,v Numeric vectors.
#' @return u minus its least-squares projection on v.
#' @noRd
.dl_residual <- function(u, v) {
  vv <- sum(v * v)
  if (vv == 0) return(u)
  u - (sum(u * v) / vv) * v
}

#' Approximate differential entropy
#'
#' Maximum-entropy approximation of Hyvarinen (1998), as used by DirectLiNGAM.
#'
#' @param u Numeric vector, expected to be standardised.
#' @return Approximate differential entropy of u.
#' @noRd
.dl_entropy <- function(u) {
  k1 <- 79.047; k2 <- 7.4129; gamma <- 0.37457
  (1 + log(2 * pi)) / 2 -
    k1 * (mean(log(cosh(u))) - gamma)^2 -
    k2 * (mean(u * exp(-u^2 / 2)))^2
}
