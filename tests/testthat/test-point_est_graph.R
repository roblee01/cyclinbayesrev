test_that("point_est_graph returns the SHD medoid", {

  A <- matrix(
    c(
      0, 1, 0,
      0, 0, 1,
      0, 0, 0
    ),
    nrow = 3,
    byrow = TRUE
  )

  B <- matrix(
    c(
      0, 1, 0,
      0, 0, 0,
      0, 0, 0
    ),
    nrow = 3,
    byrow = TRUE
  )

  posterior_graphs <- rbind(
    as.vector(A),
    as.vector(A),
    as.vector(A),
    as.vector(B)
  )

  estimate <- point_est_graph(
    posterior_graphs,
    dist_type = "shd"
  )

  expect_equal(estimate, A)
})


test_that("point_est_graph works with a custom distance", {

  A <- matrix(
    c(
      0, 1, 0,
      0, 0, 1,
      0, 0, 0
    ),
    nrow = 3,
    byrow = TRUE
  )

  B <- matrix(
    c(
      0, 1, 0,
      0, 0, 0,
      0, 0, 0
    ),
    nrow = 3,
    byrow = TRUE
  )

  posterior_graphs <- rbind(
    as.vector(A),
    as.vector(A),
    as.vector(B)
  )

  custom_distance <- function(X, Y) {
    sum(abs(X - Y))
  }

  estimate <- point_est_graph(
    posterior_graphs,
    dist_type = "custom",
    dist_fun = custom_distance
  )

  expect_equal(estimate, A)
})


test_that("point_est_graph returns the only sampled graph", {

  A <- matrix(
    c(
      0, 1, 0,
      0, 0, 1,
      0, 0, 0
    ),
    nrow = 3,
    byrow = TRUE
  )

  posterior_graphs <- matrix(
    as.vector(A),
    nrow = 1
  )

  estimate <- point_est_graph(
    posterior_graphs,
    dist_type = "shd"
  )

  expect_equal(estimate, A)
})


test_that("point_est_graph returns a square adjacency matrix", {

  p <- 4

  A <- matrix(0, p, p)
  A[1, 2] <- 1

  B <- A
  B[2, 3] <- 1

  posterior_graphs <- rbind(
    as.vector(A),
    as.vector(A),
    as.vector(B)
  )

  estimate <- point_est_graph(
    posterior_graphs,
    dist_type = "shd"
  )

  expect_equal(dim(estimate), c(p, p))
})


test_that("point_est_graph rejects an unknown distance type", {

  A <- matrix(0, 3, 3)

  posterior_graphs <- matrix(
    as.vector(A),
    nrow = 1
  )

  expect_error(
    point_est_graph(
      posterior_graphs,
      dist_type = "not_a_distance"
    )
  )
})
