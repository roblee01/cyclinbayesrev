# Tests for posterior_interval_est().
# The current implementation discards the first 75% of supplied iterations,
# uses a three-column CI matrix (lower, median, upper), and retains attributes
# attached by HDInterval::hdi() on the HPD matrix.

test_that("posterior_interval_est returns HPD and equal-tailed summaries", {
  set.seed(123)
  samples <- cbind(
    rnorm(1000, mean = -1),
    rnorm(1000, mean = 0),
    rnorm(1000, mean = 2)
  )

  result <- posterior_interval_est(samples, level = 0.95)

  expect_type(result, "list")
  expect_true(all(c("hpd_matrix", "ci_matrix") %in% names(result)))
  expect_true(is.matrix(result$hpd_matrix))
  expect_true(is.matrix(result$ci_matrix))
})


test_that("posterior_interval_est returns the correct matrix dimensions", {
  set.seed(123)
  n_parameters <- 5
  samples <- matrix(
    rnorm(1000 * n_parameters),
    nrow = 1000,
    ncol = n_parameters
  )

  result <- posterior_interval_est(samples, level = 0.95)

  expect_equal(dim(result$hpd_matrix), c(2L, n_parameters))
  # The three CI columns are lower quantile, median, and upper quantile.
  expect_equal(dim(result$ci_matrix), c(n_parameters, 3L))
  expect_equal(as.numeric(attr(result$hpd_matrix, "credMass")), 0.95)
})


test_that("posterior interval endpoints have the expected ordering", {
  set.seed(123)
  samples <- matrix(rnorm(2000), nrow = 500, ncol = 4)
  result <- posterior_interval_est(samples, level = 0.95)

  expect_true(all(result$hpd_matrix[1, ] <= result$hpd_matrix[2, ]))
  expect_true(all(result$ci_matrix[, 1] <= result$ci_matrix[, 2]))
  expect_true(all(result$ci_matrix[, 2] <= result$ci_matrix[, 3]))
})


test_that("equal-tailed summaries match empirical quantiles after 75% burn-in", {
  set.seed(123)
  samples <- matrix(rnorm(3000), nrow = 1000, ncol = 3)
  level <- 0.95
  result <- posterior_interval_est(samples, level = level)

  # Match the indexing currently used by posterior_interval_est().
  # With 1000 rows, it keeps rows 750:1000 (inclusive).
  n <- nrow(samples)
  kept <- samples[as.integer(0.75 * n):n, , drop = FALSE]
  expected <- t(apply(
    kept, 2, stats::quantile,
    probs = c((1 - level) / 2, 0.5, 1 - (1 - level) / 2),
    names = FALSE
  ))

  expect_equal(unname(result$ci_matrix), unname(expected),
               tolerance = 1e-8)
})


test_that("constant posterior draws produce constant intervals", {
  samples <- matrix(3.5, nrow = 100, ncol = 3)
  result <- posterior_interval_est(samples, level = 0.95)

  expect_equal(result$ci_matrix, matrix(3.5, nrow = 3, ncol = 3))
  # Compare HPD numeric values, not attributes added by HDInterval::hdi().
  expect_equal(dim(result$hpd_matrix), c(2L, 3L))
  expect_equal(as.numeric(result$hpd_matrix), rep(3.5, 6))
  expect_equal(as.numeric(attr(result$hpd_matrix, "credMass")), 0.95)
})


test_that("a single parameter also has a lower, median and upper CI", {
  set.seed(321)
  samples <- matrix(rnorm(400), ncol = 1)
  result <- posterior_interval_est(samples, level = 0.90)

  kept <- samples[as.integer(0.75 * nrow(samples)):nrow(samples), 1]
  expect_equal(dim(result$ci_matrix), c(1L, 3L))
  expect_equal(
    as.numeric(result$ci_matrix[1, ]),
    as.numeric(stats::quantile(kept, c(0.05, 0.5, 0.95))),
    tolerance = 1e-8
  )
  expect_equal(length(result$hpd_matrix), 2L)
})


test_that("adjacency summaries use the retained graph samples", {
  p <- 3L
  A <- matrix(0, p, p)
  B <- A
  B[2, 1] <- 1

  # With eight rows the current 75%-burn-in indexing retains rows 6:8:
  # B, A, B, so the edge 1 -> 2 has posterior inclusion probability 2/3.
  posterior_graphs <- rbind(
    as.vector(A), as.vector(A), as.vector(A), as.vector(A),
    as.vector(A), as.vector(B), as.vector(A), as.vector(B)
  )

  result <- posterior_interval_est(
    posterior_graphs,
    level = 0.95,
    adjacency = TRUE
  )

  expect_true(all(c("pip_matrix", "pip_graph_results") %in% names(result)))
  expect_equal(dim(result$pip_matrix), c(p, p))
  expect_equal(result$pip_matrix[2, 1], 2 / 3)
  expect_equal(sum(result$pip_matrix), 2 / 3)
  expect_equal(
    sort(as.numeric(result$pip_graph_results), decreasing = TRUE),
    c(2 / 3, 1 / 3),
    tolerance = 1e-8
  )
})
