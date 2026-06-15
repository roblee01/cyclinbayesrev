#' Generate synthetic DAG example data
#'
#' @description
#' Utility function to simulate data from a randomly generated sparse directed acyclic
#' graph (DAG) with non-Gaussian noise modeled as a finite Gaussian mixture.
#'
#' This is **not** an estimation or causal discovery method. It is provided solely to
#' generate toy/example datasets for demonstrations, unit tests, and simulation studies
#' used by the package's inference routines.
#'
#' @param num_covariates Integer. Number of variables (nodes) in the graph (\eqn{p}).
#' @param N Integer. Sample size (number of observations).
#' @param M_input Integer. Number of mixture components in the noise model.
#' @param prob_sparsity Numeric in (0,1). Sparsity control used when sampling edges.
#' @param seed_input Integer. Random seed for reproducibility.
#'
#' @return A list containing at least:
#' \describe{
#'   \item{data_matrix}{An \eqn{N \times p} data matrix.}
#'   \item{Adjacency_matrix_true}{A \eqn{p \times p} adjacency matrix for the true DAG.}
#' }
#'
#' @details
#' This generator is intended for internal use in examples and simulation code.
#' It should not be used as a modeling tool for real data analysis.
#'
#' @export


generates_examples_DAG = function(num_covariates,
                                  N,
                                  M_input,
                                  prob_sparsity,
                                  seed_input) {
  set.seed(seed_input)

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

  Causal_effect_matrix_true = matrix(0, num_covariates, num_covariates)
  Causal_effect_matrix_true_str = matrix(0, num_covariates, num_covariates)
  Causal_effect_matrix_true_err = matrix(0, num_covariates, num_covariates)
  Adjacency_matrix_true = matrix(0, num_covariates, num_covariates)
  identity_mat = matrix(0, num_covariates, num_covariates)
  diag(identity_mat) = 1


  rho = 1

  probability_sparsity = c(prob_sparsity, 1 - prob_sparsity)
  if (num_covariates <= 10) {
    repeat {
      probability_sparsity_tmp_1 = probability_sparsity
      Causal_effect_matrix_true_str[lower.tri(Causal_effect_matrix_true_str)] = sample(0:1,
                                                                                       length(lower.tri(Causal_effect_matrix_true_str[lower.tri(Causal_effect_matrix_true_str)])),
                                                                                       replace = TRUE,
                                                                                       prob = probability_sparsity_tmp_1)

      Causal_effect_matrix_true_str[upper.tri(Causal_effect_matrix_true_str)] = sample(0:1,
                                                                                       length(lower.tri(Causal_effect_matrix_true_str[lower.tri(Causal_effect_matrix_true_str)])),
                                                                                       replace = TRUE,
                                                                                       prob = probability_sparsity_tmp_1)
      for (i in 1:nrow(Causal_effect_matrix_true)) {
        for (j in 1:(nrow(Causal_effect_matrix_true))) {
          if (Causal_effect_matrix_true_str[i, j] &&
              Causal_effect_matrix_true_str[j, i] == 1) {
            choose_result = sample(1:2, 1, replace = TRUE)
            if (choose_result == 1) {
              Causal_effect_matrix_true_str[j, i] = 0   #sample(0:1)/100
            } else{
              Causal_effect_matrix_true_str[i, j] = 0  #sample(0:1)/100
            }
          }
        }#end of j
      }#end of i
      if (is_dag_adj(Causal_effect_matrix_true_str) &&
          (det(identity_mat - Causal_effect_matrix_true_str) != 0) &&
          (mean(Causal_effect_matrix_true_str) != 0))
        break
    }#end of repeat
  }#end of if



  Causal_effect_matrix_true = Causal_effect_matrix_true_str
  #+Causal_effect_matrix_true_err

  Adjacency_matrix_true = Causal_effect_matrix_true_str

  ## sd for epsilon
  # sigma_epsilon_true<-0.25 #since mean_tao=0.25, var_tao=0.0625


  #################################################################
  ####  Generation of epsilon_true from M mixture: M can be 1-3
  #################################################################
  #N = 500
  #truemodel=2

  data_matrix = matrix(0, N, num_covariates)

  M = 2

  ### General Case Test ###
  M_1 = 2

  mu_epsilon <- c(-0.5, 0.5)
  #mu_epsilon <- c(0,0)
  sigma_epsilon_true <- c(0.1, 0.3)

  epsilon_true <- matrix(0, N, num_covariates)
  Z_matrix_true <- matrix(0, num_covariates * N, M_1)
  # loop from 1:num_obs
  for (i_epsilon in 1:num_covariates) {
    for (z_epsilon in 1:N) {
      case_epsilon <- sample(1:M_1, size = 1, replace = T)
      ##### To express  Z^z_{i,k} as a matrix, we consider Z^z_{i,k}=Z_matrix_true[(i-1)*num_obs+z,k] #######
      iz_epsilon = (i_epsilon - 1) * N + z_epsilon
      Z_matrix_true[iz_epsilon, case_epsilon] = 1  #### row = (i_epsilon -1)*num_obs+z_epsilon

      epsilon_true[z_epsilon, i_epsilon] = rnorm(1, mu_epsilon[case_epsilon], sigma_epsilon_true[case_epsilon])
    }
  }

  ################## end of epsilon_true generation

  ########### Generation of Y using Causal_effect_matric_true and epsilon_true

  identity_mat = matrix(0, nrow = num_covariates, ncol = num_covariates)
  diag(identity_mat) = rep(1, num_covariates)

  for (i in 1:N) {
    data_matrix[i, ] =  (solve(identity_mat - Causal_effect_matrix_true) %*% epsilon_true[i, ])[, 1]
  }


  return(list(data_matrix = data_matrix, Adjacency_matrix_true = Adjacency_matrix_true))
}
