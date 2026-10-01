test_that("BayesDCG returns the expected posterior objects", {

  N <- 30
  p <- 6
  M <- 2
  num_iter <- 50
  burn_in_iterations <- 30

  example_list <- generates_examples_DCG(
    num_covariates = p,
    N = N,
    M_input = 2,
    prob_sparsity = 0.80,
    seed_input = 21,
    n_cycles = 1,
    len_range = c(2, 2),
    mag_range = c(0.4, 0.7),
    prob_positive = 0.5,
    tol_sing = 0.05,
    rho_max = 0.95
  )

  fit <- BayesDCG(
    example_list$data_matrix,
    M = M,
    num_iter = num_iter,
    burn_in_iterations = burn_in_iterations
  )

  expected_names <- c(
    "Adjacency_matrix_list",
    "Causal_effect_matrix_list",
    "gamma_list",
    "gamma_1_list",
    "mu_matrix_list",
    "tao_matrix_list",
    "pi_matrix_list",
    "log_likelihood_list",
    "n_stored"
  )

  expect_true(all(expected_names %in% names(fit)))
})


test_that("BayesDCG returns only retained posterior draws", {

  N <- 30
  p <- 6
  num_iter <- 50
  burn_in_iterations <- 30

  example_list <- generates_examples_DCG(
    num_covariates = p,
    N = N,
    M_input = 2,
    prob_sparsity = 0.80,
    seed_input = 21,
    n_cycles = 1,
    len_range = c(2, 2),
    mag_range = c(0.4, 0.7)
  )

  fit <- BayesDCG(
    example_list$data_matrix,
    M = 2,
    num_iter = num_iter,
    burn_in_iterations = burn_in_iterations
  )

  expected_n <- num_iter - burn_in_iterations

  expect_equal(fit$n_stored, expected_n)
  expect_equal(nrow(fit$Adjacency_matrix_list), expected_n)
  expect_equal(nrow(fit$Causal_effect_matrix_list), expected_n)
  expect_equal(length(fit$log_likelihood_list), expected_n)
})


test_that("BayesDCG output matrices have correct dimensions", {

  N <- 30
  p <- 6
  M <- 2

  example_list <- generates_examples_DCG(
    num_covariates = p,
    N = N,
    M_input = 2,
    prob_sparsity = 0.80,
    seed_input = 21,
    n_cycles = 1,
    len_range = c(2, 2)
  )

  fit <- BayesDCG(
    example_list$data_matrix,
    M = M,
    num_iter = 50,
    burn_in_iterations = 30
  )

  expect_equal(ncol(fit$Adjacency_matrix_list), p^2)
  expect_equal(ncol(fit$Causal_effect_matrix_list), p^2)

  expect_equal(ncol(fit$mu_matrix_list), p * M)
  expect_equal(ncol(fit$tao_matrix_list), p * M)
  expect_equal(ncol(fit$pi_matrix_list), p * M)
})


test_that("BayesDCG adjacency samples are binary with zero diagonal", {

  N <- 30
  p <- 6

  example_list <- generates_examples_DCG(
    num_covariates = p,
    N = N,
    M_input = 2,
    prob_sparsity = 0.80,
    seed_input = 21,
    n_cycles = 1,
    len_range = c(2, 2)
  )

  fit <- BayesDCG(
    example_list$data_matrix,
    M = 2,
    num_iter = 50,
    burn_in_iterations = 30
  )

  expect_true(
    all(fit$Adjacency_matrix_list %in% c(0, 1))
  )

  for (i in seq_len(nrow(fit$Adjacency_matrix_list))) {

    A <- matrix(
      fit$Adjacency_matrix_list[i, ],
      nrow = p,
      ncol = p
    )

    expect_true(all(diag(A) == 0))
  }
})


test_that("retained BayesDCG coefficient matrices satisfy stability", {

  N <- 30
  p <- 6

  example_list <- generates_examples_DCG(
    num_covariates = p,
    N = N,
    M_input = 2,
    prob_sparsity = 0.80,
    seed_input = 21,
    n_cycles = 1,
    len_range = c(2, 2),
    mag_range = c(0.4, 0.7)
  )

  fit <- BayesDCG(
    example_list$data_matrix,
    M = 2,
    num_iter = 50,
    burn_in_iterations = 30
  )

  for (i in seq_len(nrow(fit$Causal_effect_matrix_list))) {

    B <- matrix(
      fit$Causal_effect_matrix_list[i, ],
      nrow = p,
      ncol = p
    )

    rho <- max(Mod(eigen(B, only.values = TRUE)$values))

    expect_lt(rho, 1 + 1e-8)
  }
})


test_that("BayesDCG rejects incorrectly sized initial matrices", {

  N <- 30
  p <- 6

  example_list <- generates_examples_DCG(
    num_covariates = p,
    N = N,
    M_input = 2,
    prob_sparsity = 0.80,
    seed_input = 21,
    n_cycles = 1,
    len_range = c(2, 2)
  )

  bad_init <- matrix(0, 3, 3)

  expect_error(
    BayesDCG(
      example_list$data_matrix,
      M = 2,
      num_iter = 20,
      burn_in_iterations = 10,
      init_Adjacency = bad_init
    )
  )
})
