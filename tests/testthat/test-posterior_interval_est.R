test_that("posterior_interval_est returns HPD and equal-tailed intervals", {

  set.seed(123)

  samples <- cbind(
    rnorm(1000, mean = -1),
    rnorm(1000, mean = 0),
    rnorm(1000, mean = 2)
  )

  result <- posterior_interval_est(
    samples,
    level = 0.95
  )

  expect_true(is.list(result))

  expect_true(
    all(c("hpd_matrix", "ci_matrix") %in% names(result))
  )
})


test_that("posterior_interval_est returns the expected dimensions", {

  set.seed(123)

  n_parameters <- 5

  samples <- matrix(
    rnorm(1000 * n_parameters),
    nrow = 1000,
    ncol = n_parameters
  )

  result <- posterior_interval_est(
    samples,
    level = 0.95
  )

  expect_equal(
    dim(result$hpd_matrix),
    c(2, n_parameters)
  )

  expect_equal(
    dim(result$ci_matrix),
    c(n_parameters, 2)
  )
})


test_that("posterior interval lower bounds do not exceed upper bounds", {

  set.seed(123)

  samples <- matrix(
    rnorm(2000),
    nrow = 500,
    ncol = 4
  )

  result <- posterior_interval_est(
    samples,
    level = 0.95
  )

  expect_true(
    all(
      result$hpd_matrix[1, ] <=
        result$hpd_matrix[2, ]
    )
  )

  expect_true(
    all(
      result$ci_matrix[, 1] <=
        result$ci_matrix[, 2]
    )
  )
})


test_that("equal-tailed intervals agree with empirical quantiles", {

  set.seed(123)

  samples <- matrix(
    rnorm(3000),
    nrow = 1000,
    ncol = 3
  )

  result <- posterior_interval_est(
    samples,
    level = 0.95
  )

  expected <- t(
    apply(
      samples,
      2,
      stats::quantile,
      probs = c(0.025, 0.975)
    )
  )

  expect_equal(
    result$ci_matrix,
    expected,
    tolerance = 1e-8
  )
})


test_that("constant posterior samples give zero-width intervals", {

  samples <- matrix(
    3.5,
    nrow = 100,
    ncol = 3
  )

  result <- posterior_interval_est(
    samples,
    level = 0.95
  )

  expect_equal(
    result$ci_matrix,
    matrix(3.5, nrow = 3, ncol = 2)
  )

  expect_equal(
    result$hpd_matrix,
    matrix(3.5, nrow = 2, ncol = 3)
  )
})
