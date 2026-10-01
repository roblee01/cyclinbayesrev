test_that("BayesDAG returns the expected posterior objects", {

  N <- 30
  p <- 4
  M <- 2
  num_iter <- 50
  burn_in_iterations <- 10

  example_list <- generates_examples_DAG(
    num_covariates = p,
    N = N,
    M_input = 2,
    prob_sparsity = 0.80,
    seed_input = 21,
    mag_range = c(0.4, 0.9),
    prob_positive = 0.5,
    seed_structure = 1,
    seed_weights = 20
  )

  fit <- BayesDAG(
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
    "first_stored_iteration",
    "n_stored"
  )

  expect_true(all(expected_names %in% names(fit)))
})


test_that("BayesDAG returns the correct number of retained draws", {

  N <- 30
  p <- 4
  M <- 2
  num_iter <- 50
  burn_in_iterations <- 10

  example_list <- generates_examples_DAG(
    num_covariates = p,
    N = N,
    M_input = 2,
    prob_sparsity = 0.80,
    seed_input = 21,
    seed_structure = 1,
    seed_weights = 20
  )

  fit <- BayesDAG(
    example_list$data_matrix,
    M = M,
    num_iter = num_iter,
    burn_in_iterations = burn_in_iterations
  )

  expected_n <- num_iter - burn_in_iterations

  expect_equal(fit$n_stored, expected_n)
  expect_equal(nrow(fit$Adjacency_matrix_list), expected_n)
  expect_equal(nrow(fit$Causal_effect_matrix_list), expected_n)
  expect_equal(length(fit$gamma_list), expected_n)
  expect_equal(length(fit$gamma_1_list), expected_n)
  expect_equal(length(fit$log_likelihood_list), expected_n)

  expect_equal(
    fit$first_stored_iteration,
    burn_in_iterations + 1
  )
})


test_that("BayesDAG output matrices have the correct dimensions", {

  N <- 30
  p <- 4
  M <- 2

  example_list <- generates_examples_DAG(
    num_covariates = p,
    N = N,
    M_input = 2,
    prob_sparsity = 0.80,
    seed_input = 21,
    seed_structure = 1,
    seed_weights = 20
  )

  fit <- BayesDAG(
    example_list$data_matrix,
    M = M,
    num_iter = 50,
    burn_in_iterations = 10
  )

  expect_equal(ncol(fit$Adjacency_matrix_list), p^2)
  expect_equal(ncol(fit$Causal_effect_matrix_list), p^2)

  expect_equal(ncol(fit$mu_matrix_list), p * M)
  expect_equal(ncol(fit$tao_matrix_list), p * M)
  expect_equal(ncol(fit$pi_matrix_list), p * M)
})


test_that("BayesDAG adjacency samples are binary with zero diagonal", {

  N <- 30
  p <- 4

  example_list <- generates_examples_DAG(
    num_covariates = p,
    N = N,
    M_input = 2,
    prob_sparsity = 0.80,
    seed_input = 21,
    seed_structure = 1,
    seed_weights = 20
  )

  fit <- BayesDAG(
    example_list$data_matrix,
    M = 2,
    num_iter = 50,
    burn_in_iterations = 10
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


test_that("BayesDAG automatically removes the annealing period", {

  N <- 30
  p <- 4
  num_iter <- 50

  example_list <- generates_examples_DAG(
    num_covariates = p,
    N = N,
    M_input = 2,
    prob_sparsity = 0.80,
    seed_input = 21,
    seed_structure = 1,
    seed_weights = 20
  )

  fit <- BayesDAG(
    example_list$data_matrix,
    M = 2,
    num_iter = num_iter,
    burn_in_iterations = 0
  )

  # Minimum effective burn-in is 20% of 50 = 10
  expect_equal(fit$first_stored_iteration, 11)
  expect_equal(fit$n_stored, 40)
  expect_equal(nrow(fit$Adjacency_matrix_list), 40)
})


test_that("BayesDAG rejects a cyclic initial adjacency matrix", {

  N <- 30
  p <- 4

  example_list <- generates_examples_DAG(
    num_covariates = p,
    N = N,
    M_input = 2,
    prob_sparsity = 0.80,
    seed_input = 21,
    seed_structure = 1,
    seed_weights = 20
  )

  cyclic_init <- matrix(0, p, p)

  # 1 -> 2 and 2 -> 1
  cyclic_init[2, 1] <- 1
  cyclic_init[1, 2] <- 1

  expect_error(
    BayesDAG(
      example_list$data_matrix,
      M = 2,
      num_iter = 20,
      burn_in_iterations = 4,
      init_Adjacency = cyclic_init
    )
  )
})
