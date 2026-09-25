# Helpers for optional initial states passed from BayesDAG()/BayesDCG()
# to the Rcpp samplers. NULL means "let the C++ sampler initialize this".

.init_matrix <- function(x, name) {
  if (is.null(x)) return(NULL)

  x <- as.matrix(x)
  if (!is.numeric(x) || is.complex(x) || anyNA(x) || any(!is.finite(x))) {
    stop(sprintf("%s must be a finite numeric matrix or NULL.", name),
         call. = FALSE)
  }
  storage.mode(x) <- "double"
  x
}

# The C++ init_gamma_1 and init_gamma_result parameters are nullable
# NumericVector arguments; R scalar doubles work for each of them.
.init_scalar <- function(x, name) {
  if (is.null(x)) return(NULL)

  if (!is.numeric(x) || is.complex(x) || length(x) != 1L ||
      is.na(x) || !is.finite(x)) {
    stop(sprintf("%s must be one finite numeric value or NULL.", name),
         call. = FALSE)
  }
  as.numeric(x)
}

# Optional synonym in case a wrapper calls .init_vector for a numeric
# initial-state vector rather than .init_scalar.
.init_vector <- function(x, name) {
  if (is.null(x)) return(NULL)

  if (!is.numeric(x) || is.complex(x) || length(x) < 1L ||
      anyNA(x) || any(!is.finite(x))) {
    stop(sprintf("%s must be a nonempty finite numeric vector or NULL.", name),
         call. = FALSE)
  }
  as.numeric(x)
}
