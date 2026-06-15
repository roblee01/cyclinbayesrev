#' Posterior Interval Estimate for MCMC samples
#'
#' @description
#' Computes posterior summaries from a matrix of MCMC samples. When
#' \code{adjacency = TRUE}, each row is treated as a vectorized (flattened)
#' adjacency matrix and the function returns edge-wise posterior inclusion
#' probabilities and posterior frequencies of unique graph structures (computed
#' after discarding the first 75 percent of iterations). When \code{adjacency = FALSE},
#' each column is treated as a scalar parameter and the function returns equal-tailed
#' credible intervals and HPD intervals at the specified credible level (also using
#' the last 25 percent of iterations).
#'
#' @param posterior_matrix Numeric matrix of posterior samples. Rows correspond to
#'   MCMC iterations. If \code{adjacency = TRUE}, columns must represent a flattened
#'   \eqn{p \times p} adjacency matrix.
#' @param level Numeric in (0,1). Credible level for interval summaries (only used when
#'   \code{adjacency = FALSE}).
#' @param adjacency Logical. If \code{TRUE}, compute adjacency-specific summaries
#'   (posterior inclusion probabilities and graph-structure frequencies). If \code{FALSE},
#'   compute interval summaries for each column of \code{posterior_matrix}.
#'
#' @return
#' If \code{adjacency = TRUE}, a list with components:
#' \describe{
#'   \item{pip_matrix}{Numeric \eqn{p \times p} matrix of edge-wise posterior inclusion
#'   probabilities. Entry \eqn{(i,j)} is the proportion of retained draws containing an edge
#'   from node \eqn{i} to node \eqn{j}.}
#'   \item{pip_graph_results}{Named vector/table giving posterior frequencies of unique
#'   graph structures among the retained draws.}
#' }
#'
#' If \code{adjacency = FALSE}, a list with components:
#' \describe{
#'   \item{hpd_matrix}{Numeric matrix of HPD intervals for each parameter/column.}
#'   \item{ci_matrix}{Numeric matrix of equal-tailed credible intervals (lower, median, upper)
#'   for each parameter/column.}
#' }
#'
#' @export
#'
#' @examples
#'
#' # Run BayesDAG simulated example
#'
#' N = 300
#' num_covariates = 10
#' M = 2
#' num_iter = 1000
#'
#'
#' example_list = generates_examples_DAG(num_covariates, N, M, 0.9, 21)
#'
#' data_matrix = example_list$data_matrix
#' Adjacency_matrix_true = example_list$Adjacency_matrix_true
#'
#' params = list(
#' a_mu = 0,
#' b_mu = 2,
#' a_gamma = 0.5,
#' b_gamma = 0.5,
#' a_gamma_1 = 2,
#' b_gamma_1 = 1,
#' a_tao = 2,
#' b_tao = 1,
#' a_og_tao = 0.01,
#' b_og_tao = 0.01,
#' alpha = 1
#' )
#'
#'
#' result_list = BayesDAG(
#' data_matrix,
#' params$a_mu,
#' params$b_mu,
#' params$a_gamma,
#' params$b_gamma,
#' params$a_tao,
#' params$b_tao,
#' params$a_og_tao,
#' params$b_og_tao,
#' params$a_gamma_1,
#' params$b_gamma_1,
#' params$alpha,
#' M,
#' num_iter
#' ) # Runs the Acyclic algorithm
#'
#'
#' # Extract posterior outputs
#' Adjacency_matrix_means = result_list$Adjacency_matrix_means
#' Adjacency_matrix_list = result_list$Adjacency_matrix_list
#' Causal_effect_matrix_list = result_list$Causal_effect_matrix_list
#' gamma_list = result_list$gamma_list
#' gamma_1_list = result_list$gamma_1_list
#' mu_matrix_list = result_list$mu_matrix_list
#' tao_matrix_list = result_list$tao_matrix_list
#' pi_matrix_list = result_list$pi_matrix_list
#'
#'Causal_effect_matrix_summary = posterior_interval_est(Causal_effect_matrix_list, level = 0.95)
#'hpd_matrix_acyclic = Causal_effect_matrix_summary$hpd_matrix
#'ci_matrix_acyclic = Causal_effect_matrix_summary$ci_matrix
#'
#'#######################################
#'# Extracting nonzero HPD intervals
#'#######################################
#'par(mfrow=c(2,1))
#'
#'nonzero_cols = which(colSums(hpd_matrix_acyclic) != 0)
#'num_non_zero_coef = length(nonzero_cols)
#'data_1 = data.frame(cbind(1:num_non_zero_coef,t(hpd_matrix_acyclic[,which(colSums(hpd_matrix_acyclic)!=0)])))
#'
#'nonzero_cols = which(colSums(hpd_matrix_acyclic) != 0)
#'
#'# subset and transpose so each row = coefficient
#'hpd_sub = t(hpd_matrix_acyclic[, nonzero_cols, drop = FALSE])
#'colnames(hpd_sub) = c("lower", "upper")  # row1 = lower, row2 = upper
#'
#'data_1 = as.data.frame(hpd_sub)
#'data_1$x = factor(seq_len(nrow(data_1)))
#'
#'data_1$mid = (data_1$lower + data_1$upper) / 2
#'
#'
#'ggplot2::ggplot(data_1, ggplot2::aes(x = x, y = mid)) +
#'  ggplot2::geom_point(size = 3) +
#'  ggplot2::geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
#'  ggplot2::geom_errorbar(ggplot2::aes(ymin = lower, ymax = upper), width = 0.2) +
#'  ggplot2::labs(y = "Causal Weight Estimate with HPD Interval", x = "Nonzero causal effect coefficient (index)") +
#'  ggplot2::theme_minimal()
#'
#'
#'#######################################
#'# Extracting nonzero Credible intervals
#'#######################################
#'data_2 = data.frame(ci_matrix_acyclic[which(rowSums(ci_matrix_acyclic)!=0),])
#'x = factor(1:nrow(data_2))
#'data_2 = cbind(x,data_2)
#'
#' ggplot2::ggplot(data_2, ggplot2::aes(x = x, y = X2)) +
#'   ggplot2::geom_point(size = 3) +
#'   ggplot2::geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
#'   ggplot2::geom_errorbar(ggplot2::aes(ymin = X1, ymax = X3), width = 0.2) +  # just X1/X3
#'   ggplot2::labs(y = "Causal Weight Estimate with 95% CI", x = "Nonzero causal effect coefficient (index)") +
#'   ggplot2::theme_minimal()

posterior_interval_est = function(posterior_matrix, level, adjacency = FALSE){

  num_iter = nrow(posterior_matrix)

  if(adjacency){
    num_iter = nrow(posterior_matrix)
    num_features = sqrt(ncol(posterior_matrix))
    Adjacency_matrix_list_75 = posterior_matrix[(0.75*num_iter):num_iter,]
    pip_matrix = matrix(colMeans(Adjacency_matrix_list_75), num_features, num_features)


    denominator_val = nrow(Adjacency_matrix_list_75)
    probabilities_vec = rep(0,denominator_val)
    for(i in 1:denominator_val){
      probabilities_vec[i] = paste(Adjacency_matrix_list_75[i,],collapse='')
    }
    pip_graph_results = sort(table(probabilities_vec),decreasing=TRUE)/denominator_val

    return(list(pip_matrix = pip_matrix, pip_graph_results = pip_graph_results))
  }

  ci_matrix = matrix(0,ncol(posterior_matrix),3)

  if(ncol(posterior_matrix) == 1){
    posterior_75 = as.matrix(posterior_matrix[(0.75*num_iter):num_iter,])
  } else{
    posterior_75 = posterior_matrix[(0.75*num_iter):num_iter,]
  }

  hpd_matrix = HDInterval::hdi(posterior_75, credMass = level)

  for(i in 1:ncol(posterior_matrix)){
    ci_matrix[i,] =  quantile(posterior_75[,i],c((1-level)/2,0.5,1-(1-level)/2))
  }

  return(list(hpd_matrix = hpd_matrix, ci_matrix = ci_matrix))
}
