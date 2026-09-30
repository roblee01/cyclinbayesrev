test_that("BayesDAG returns correctly shaped posterior samples", {

  set.seed(123)

  N = 30
  p = 3
  M = 2
  num_iter = 50
  burn_in_iterations = 10

  data_matrix = matrix(
    rnorm(N * p),
    nrow = N,
    ncol = p
  )

  fit = BayesDAG(
    data_matrix,
    a_mu = 0,
    b_mu = 2,
    a_gamma = 0.5,
    b_gamma = 0.5,
    a_tao = 2,
    b_tao = 1,
    a_og_tao = 0.01,
    b_og_tao = 0.01,
    a_gamma_1 = 2,
    b_gamma_1 = 1,
    alpha = 1,
    M = M,
    num_iter = num_iter,
    burn_in_iterations = burn_in_iterations
  )

  expect_true(is.list(fit))

  expect_equal(
    nrow(fit$Adjacency_matrix_list),
    num_iter - burn_in_iterations
  )

  expect_equal(
    ncol(fit$Adjacency_matrix_list),
    p^2
  )

  expect_equal(
    nrow(fit$Causal_effect_matrix_list),
    num_iter - burn_in_iterations
  )

  expect_equal(
    ncol(fit$Causal_effect_matrix_list),
    p^2
  )

  expect_equal(
    fit$first_stored_iteration,
    burn_in_iterations + 1
  )

  expect_equal(
    fit$n_stored,
    num_iter - burn_in_iterations
  )
})

test_that("BayesDAG removes annealing-period draws automatically", {

  set.seed(123)

  N = 30
  p = 3
  M = 2
  num_iter = 50

  data_matrix = matrix(
    rnorm(N * p),
    nrow = N,
    ncol = p
  )

  fit = BayesDAG(
    data_matrix,
    M = M,
    num_iter = num_iter,
    burn_in_iterations = 0
  )

  # 20% of 50 = 10 iterations discarded
  expect_equal(
    fit$first_stored_iteration,
    11
  )

  expect_equal(
    fit$n_stored,
    40
  )

  expect_equal(
    nrow(fit$Adjacency_matrix_list),
    40
  )

  expect_equal(
    length(fit$log_likelihood_list),
    40
  )
})
