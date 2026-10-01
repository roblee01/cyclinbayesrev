# Tests for posterior_network_motif().
# Each posterior adjacency matrix is flattened column-by-column with c(A),
# matching the vectorisation used by the samplers and the implementation.

.make_motif_graph <- function(A) {
  igraph::graph_from_adjacency_matrix(
    A,
    mode = "directed",
    diag = FALSE
  )
}

.make_motif_samples <- function(...) {
  graphs <- list(...)
  do.call(rbind, lapply(graphs, as.vector))
}

test_that("a motif present in every posterior draw has frequency one", {
  A <- matrix(0, 3, 3)
  A[1, 2] <- 1
  A[2, 3] <- 1

  posterior <- .make_motif_samples(A, A, A)
  observed <- posterior_network_motif(.make_motif_graph(A), posterior)

  expect_equal(observed, 1)
})

test_that("a motif absent from every posterior draw has frequency zero", {
  motif <- matrix(0, 3, 3)
  motif[1, 2] <- 1
  motif[2, 3] <- 1

  A <- matrix(0, 3, 3)
  A[1, 2] <- 1  # Missing the second required edge.

  posterior <- .make_motif_samples(A, A, A)
  observed <- posterior_network_motif(.make_motif_graph(motif), posterior)

  expect_equal(observed, 0)
})

test_that("motif frequency is the fraction containing every motif edge", {
  motif <- matrix(0, 3, 3)
  motif[1, 2] <- 1
  motif[2, 3] <- 1

  exact <- motif
  extra <- motif
  extra[1, 3] <- 1  # A supergraph also contains the motif.

  incomplete <- motif
  incomplete[2, 3] <- 0

  posterior <- .make_motif_samples(exact, extra, incomplete)
  observed <- posterior_network_motif(.make_motif_graph(motif), posterior)

  expect_equal(observed, 2 / 3)
})

test_that("extra posterior edges do not count against motif inclusion", {
  motif <- matrix(0, 3, 3)
  motif[1, 2] <- 1

  A <- motif
  A[2, 3] <- 1

  B <- motif
  B[3, 1] <- 1

  posterior <- .make_motif_samples(A, B)
  observed <- posterior_network_motif(.make_motif_graph(motif), posterior)

  # Neither sampled graph is exactly equal to the motif.
  expect_false(identical(A, motif))
  expect_false(identical(B, motif))
  expect_equal(observed, 1)
})

test_that("edge direction matters for directed motifs", {
  motif <- matrix(0, 3, 3)
  motif[1, 2] <- 1

  reversed <- matrix(0, 3, 3)
  reversed[2, 1] <- 1

  posterior <- .make_motif_samples(motif, reversed)
  observed <- posterior_network_motif(.make_motif_graph(motif), posterior)

  expect_equal(observed, 0.5)
})

test_that("a single posterior draw is handled correctly", {
  motif <- matrix(0, 3, 3)
  motif[1, 2] <- 1

  absent <- matrix(0, 3, 3)

  # drop = FALSE preserves the required sample-matrix input shape.
  posterior_present <- matrix(as.vector(motif), nrow = 1)
  posterior_absent <- matrix(as.vector(absent), nrow = 1)

  expect_equal(
    posterior_network_motif(.make_motif_graph(motif), posterior_present),
    1
  )
  expect_equal(
    posterior_network_motif(.make_motif_graph(motif), posterior_absent),
    0
  )
})

test_that("an empty-edge motif is present in every posterior graph", {
  motif <- matrix(0, 3, 3)
  A <- matrix(0, 3, 3)
  A[1, 2] <- 1
  B <- matrix(0, 3, 3)
  B[2, 3] <- 1

  posterior <- .make_motif_samples(A, B)
  observed <- posterior_network_motif(.make_motif_graph(motif), posterior)

  # all(integer(0) %in% anything) is TRUE, as used by the implementation.
  expect_equal(observed, 1)
})

test_that("directed cyclic motifs are supported", {
  cycle <- matrix(0, 3, 3)
  cycle[1, 2] <- 1
  cycle[2, 3] <- 1
  cycle[3, 1] <- 1

  broken_cycle <- cycle
  broken_cycle[3, 1] <- 0

  posterior <- .make_motif_samples(cycle, broken_cycle)
  observed <- posterior_network_motif(.make_motif_graph(cycle), posterior)

  expect_equal(observed, 0.5)
})

test_that("all supplied posterior draws contribute to motif frequency", {
  motif <- matrix(0, 3, 3)
  motif[1, 2] <- 1
  absent <- matrix(0, 3, 3)

  # The implementation does NOT discard an additional burn-in fraction.
  posterior <- .make_motif_samples(motif, absent, absent, absent)
  observed <- posterior_network_motif(.make_motif_graph(motif), posterior)

  expect_equal(observed, 0.25)
})
