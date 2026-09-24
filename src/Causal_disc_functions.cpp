#include <RcppArmadillo.h>
#include <queue>
using namespace Rcpp;

// Bayesian causal discovery samplers.
//   BCD_v2_two_phase_cpp  - cyclic graphs (disjoint directed cycles)
//   BayesSCLingam_cpp     - acyclic graphs (DAGs)
// Both model the error terms as a Dirichlet-process-style finite mixture of
// normals, so the models are identified without a Gaussian assumption.

// Random draws not provided by Armadillo directly.

arma::vec rbeta_cpp(int n, double alpha, double beta) {
  arma::vec g1 = arma::randg<arma::vec>(n, arma::distr_param(alpha, 1.0));
  arma::vec g2 = arma::randg<arma::vec>(n, arma::distr_param(beta, 1.0));
  arma::vec result = (g1 / (g1 + g2)).eval();
  return result;
}

// [[Rcpp::export("rdirichlet_cpp")]]
arma::vec rdirichlet_cpp(const arma::vec& alpha) {
  int k = alpha.n_elem;
  arma::vec y(k);
  for (int i = 0; i < k; i++) {
    y(i) = randg(arma::distr_param(alpha(i), 1.0));
  }

  return y / arma::sum(y);
}

// [[Rcpp::export("rinvgamma_cpp")]]
arma::vec rinvgamma_cpp(arma::uword n, double shape, double scale) {
  arma::vec gamma_samples = arma::randg<arma::vec>(n, arma::distr_param(shape, 1.0 / scale));
  return 1.0 / gamma_samples;
}

int sample_categorical_cpp(const arma::rowvec& probs) {
  arma::rowvec norm_probs = probs / arma::accu(probs);
  arma::rowvec cumprobs = arma::cumsum(norm_probs);
  double u = arma::randu();

  for (arma::uword i = 0; i < cumprobs.n_elem; ++i) {
    if (u < cumprobs[i]) {
      return i;
    }
  }
  return probs.n_elem-1;
}

double log_dmvn(const arma::rowvec& x, const arma::colvec& mu, const arma::mat& Sigma) {
  const double log2pi = std::log(2.0 * M_PI);
  arma::vec diff = arma::conv_to<arma::vec>::from(x.t()) - mu; // make column
  arma::mat L;
  bool status = arma::chol(L, Sigma, "lower"); // Sigma = L * L.t()
  if (!status) {
    return -arma::datum::inf; // singular; return very small log-prob
  }
  arma::vec sol = arma::solve(arma::trimatl(L), diff);          // L * y = diff
  arma::vec quad = arma::solve(arma::trimatu(L.t()), sol);      // L.t() * z = y -> z = Sigma^{-1} diff
  double maha = arma::dot(sol, sol);                           // diff' * inv(Sigma) * diff = ||sol||^2
  double logDet = 2.0 * arma::accu(arma::log(arma::diagvec(L))); // log |Sigma|
  int d = x.n_elem;
  return -0.5 * (d * log2pi + logDet + maha);
}

inline double fast_dnorm_log(double x, double mean, double sd) {
  const double log_sqrt_2pi = 0.9189385332046727;
  double z = (x - mean) / sd;
  return -log_sqrt_2pi - std::log(sd) - 0.5 * z * z;
}

inline arma::vec fast_dnorm_log_vec(const arma::vec& x,
                                    double mean,
                                    double sd) {
  const double log_sqrt_2pi = 0.9189385332046727;
  arma::vec z = (x - mean) / sd;              // elementwise
  return -log_sqrt_2pi
  - std::log(sd)
    - 0.5 * arma::square(z);            // elementwise square
}

// [[Rcpp::export("logSumExp_v2")]]
double logSumExp(const arma::rowvec& x) {
  double max_val = x.max();
  return max_val + log(sum(exp(x - max_val)));
}

// [[Rcpp::export("log_dgamma_v2")]]
double log_dgamma(double x, double a, double b) {
  return a * std::log(b) - std::lgamma(a) - (a + 1.0) * std::log(x) - b / x;
}

// Mixture-component Gibbs updates

// [[Rcpp::export]]
arma::mat mu_fun(const arma::mat& Z_matrix, double a_mu, double b_mu,
                 const arma::mat& tao_mat, const arma::mat& epsilon_mat,
                 int num_covariates, int M, int N){
  // NOTE: a_mu and b_mu are double here, NOT int as in the original.
  arma::vec numerator_result(M, arma::fill::zeros);
  arma::mat mu_mat(num_covariates, M, arma::fill::zeros);

  for(int i = 0; i < num_covariates; i++){
    arma::mat first_part = Z_matrix.rows(i * N, (i + 1) * N - 1);
    first_part.each_row() /= tao_mat.row(i);

    for(int j = 0; j < M; j++){
      numerator_result(j) = (a_mu / b_mu)
      + arma::dot(first_part.col(j), epsilon_mat.col(i));
    }

    arma::vec denominator_result = (1.0 / b_mu) + arma::sum(first_part, 0).t();

    arma::vec mean_vec = numerator_result / denominator_result;
    arma::vec sd_vec   = arma::sqrt(1.0 / denominator_result);

    arma::rowvec mu_row = mean_vec.t() + arma::randn<arma::rowvec>(M) % sd_vec.t();
    mu_mat.row(i) = mu_row;
  }
  return mu_mat;
}

// [[Rcpp::export]]
arma::mat tao_fun(const arma::mat& Z_matrix, double a_tao, double b_tao,
                  const arma::mat& mu_mat, const arma::mat& epsilon_mat,
                  int num_covariates, int M, int N) {

  arma::mat tao_mat(num_covariates, M, arma::fill::zeros);

  for (int i = 0; i < num_covariates; i++) {
    arma::mat Z_portion = Z_matrix.rows(i * N, (i + 1) * N - 1);
    arma::rowvec a = a_tao + arma::sum(Z_portion, 0) / 2;

    for (int j = 0; j < M; j++) {
      arma::vec eps_col       = epsilon_mat.col(i);
      double    mu_ij         = mu_mat(i, j);
      arma::vec squared_diff  = arma::square(eps_col - mu_ij);
      arma::vec b_portion     = 0.5 * Z_portion.col(j) % squared_diff;
      double    b             = b_tao + arma::sum(b_portion);

      arma::vec tao_sample = rinvgamma_cpp(1, a(j), b);
      tao_mat(i, j) = tao_sample(0);
    }
  }

  return tao_mat;
}

// [[Rcpp::export]]
arma::mat pi_fun(const arma::mat& Z_matrix, int num_covariates,
                 double alpha, double K, int N){

  arma::mat pi_mat(static_cast<arma::uword>(num_covariates),
                   static_cast<arma::uword>(K));

  for(int i = 0; i < num_covariates; i++){
    arma::mat Z_portion = Z_matrix.rows(i * N, (i + 1) * N - 1);

    arma::rowvec Z_colsum  = arma::sum(Z_portion, 0);
    arma::vec    alpha_vec = Z_colsum.t() + alpha;

    pi_mat.row(i) = arma::trans(rdirichlet_cpp(alpha_vec));
  }
  return pi_mat;
}

// [[Rcpp::export]]
arma::mat Z_matrix_fun(arma::mat Z_matrix_1, arma::mat epsilon_mat,
                       arma::mat mu_mat, arma::mat tao_mat, arma::mat pi_mat,
                       double num_covariates, int N, int M){

  arma::mat practice_prob_mat(N, M);
  arma::mat Z_matrix = Z_matrix_1;

  for(int i = 0; i < num_covariates; i++){
    for(int z = 0; z < N; z++){
      int iz = i * N + z;

      Z_matrix.row(iz).zeros();

      for(int j = 0; j < M; j++){
        double mu_1  = mu_mat(i, j);
        double tao_1 = tao_mat(i, j);
        practice_prob_mat(z, j) =
          fast_dnorm_log(epsilon_mat(z, i), mu_1, std::sqrt(tao_1));
      }

      arma::rowvec log_probs = arma::log(pi_mat.row(i)) + practice_prob_mat.row(z);
      arma::rowvec probs     = arma::exp(log_probs - logSumExp(log_probs));

      double u        = arma::randu();
      double cum_prob = 0.0;
      int    sampled  = M - 1;

      for (int m = 0; m < M; m++) {
        cum_prob += probs[m];
        if (u < cum_prob) {
          sampled = m;
          break;
        }
      }

      Z_matrix(iz, sampled) = 1;
    }
  }

  return Z_matrix;
}

// DAG Function

// [[Rcpp::export("is_dag")]]
bool is_dag(const arma::mat& adj){
  arma::uword n = adj.n_rows;

  // Use an integer vector for in-degree counts.
  arma::Col<int> in_degree(n);
  for (arma::uword col = 0; col < n; ++col) {
    // sum of column = in-degree of node col
    in_degree(col) = static_cast<int>(arma::accu(adj.col(col)));
  }

  std::queue<arma::uword> q;
  for (arma::uword i = 0; i < n; ++i) {
    if (in_degree(i) == 0) {
      q.push(i);
    }
  }

  int count = 0;
  while (!q.empty()) {
    arma::uword u = q.front();
    q.pop();
    ++count;

    for (arma::uword v = 0; v < n; ++v) {
      if (adj(u, v) != 0) {
        in_degree(v)--;
        if (in_degree(v) == 0) {
          q.push(v);
        }
      }
    }
  }

  return count == static_cast<int>(n);
}

bool is_positive_definite(const arma::mat& A) {
  if (!arma::approx_equal(A, A.t(), "absdiff", 1e-10)) {
    return false;
  }
  arma::mat L;
  return arma::chol(L, A);
}

double log_integral_result_calculator_cpp(const arma::mat& Adjacency_matrix_enter, const arma::mat& Z_matrix, const arma::mat& data_matrix, double current_y, const arma::vec& tao_input, bool need_tao, int N, int M, double gamma_1, double b_mu, double a_tao, double b_tao, double which_iter, double num_iter){
  arma::mat Z_current = Z_matrix.rows(static_cast<arma::uword>(current_y-1)*static_cast<arma::uword>(N),(static_cast<arma::uword>(current_y-1)+1)*static_cast<arma::uword>(N)-1);

  arma::vec Z_vec = arma::vectorise(Z_current.t());
  std::vector<int> current_assignments_vec;
  // Rcpp::IntegerVector tmp = Rcpp::wrap(Z_vec);
  for(arma::uword i = 0; i < Z_vec.n_elem; i++){
    if(Z_vec[i] == 1){
      int index = (i+1) % M;

      if(index == 0){
        index = M;
      }
      current_assignments_vec.push_back(index);
    }
  }

  arma::uvec indices(static_cast<arma::uword>(current_assignments_vec.size()));
  for(arma::uword i = 0; i < current_assignments_vec.size(); i++){
    indices[i] = current_assignments_vec[i] - 1;
  }

  arma::vec current_tao_list = tao_input.elem(indices);

  arma::uvec selected_cols = arma::find(Adjacency_matrix_enter.row(static_cast<arma::uword>(current_y-1)) == 1);
  arma::mat used_ys = data_matrix.cols(selected_cols);

  double ncol_used_ys = used_ys.n_cols;

  arma::mat used_ys_plus = arma::join_rows(used_ys, Z_current);

  arma::mat weighted_input = used_ys_plus.each_col() / current_tao_list;
  arma::mat V_mat = trans(weighted_input) * used_ys_plus;

  arma::vec diag_correction_1(static_cast<arma::uword>(ncol_used_ys));
  diag_correction_1.fill(1/gamma_1);

  arma::vec diag_correction_2(M);
  diag_correction_2.fill(1/b_mu);

  arma::vec diag_correction = join_cols(diag_correction_1,diag_correction_2);

  V_mat.diag() += diag_correction;

  double tao_density_portion = 0;
  if(need_tao){
    for(arma::uword k = 0; k < tao_input.n_elem; k++){
      double x = tao_input(k);
      tao_density_portion += log_dgamma(x, a_tao, b_tao);
    }
  }
  double first_part = -(N + M + ncol_used_ys) * std::log(2 *arma::datum::pi) + arma::sum(arma::log(1/arma::sqrt(current_tao_list))) + arma::sum(arma::log(arma::sqrt(diag_correction)));
  double first_part_1 = (1 - std::exp(-20 * which_iter / num_iter)) * (N / 2.0 + M + selected_cols.n_elem) * std::log(2 * M_PI);

  arma::mat L_chol;
  bool is_spd = arma::chol(L_chol, V_mat, "lower");  // V_mat = L_chol * L_chol.t()

  double logdet;
  arma::mat V_inv;

  if(is_spd){
    // log|V_mat| from the Cholesky factor — avoids a separate LU-based
    // log_det
    logdet = 2.0 * arma::accu(arma::log(arma::diagvec(L_chol)));
    // inverse from the same factor — avoids inv_sympd redoing the
    // decomposition
    arma::mat L_inv = arma::inv(arma::trimatl(L_chol));
    V_inv = L_inv.t() * L_inv;
  } else {
    double sign;
    arma::log_det(logdet, sign, V_mat);
    V_inv = arma::pinv(V_mat);
  }

  double second_part = -0.5 * logdet - 0.5 * sum((1 / current_tao_list) % square(data_matrix.col(static_cast<arma::uword>(current_y-1))));

  arma::vec weight_vec = data_matrix.col(current_y-1) % (1 / current_tao_list);
  arma::rowvec third_part_1 = trans(weight_vec) * used_ys_plus;

  double third_part = (1 + std::exp(-20 * which_iter / num_iter)) * as_scalar(third_part_1 * V_inv * third_part_1.t());
  if(need_tao){
    return first_part + first_part_1 + second_part + 0.5 * third_part + tao_density_portion;
  } else{
    return first_part + first_part_1 + second_part + 0.5 * third_part;
  }
}

// Precompute, once, an N x p matrix giving each data point's currently
// assigned mixture-component mean/variance for every feature.
void compute_mixture_mean_var(const arma::mat& Z_matrix, const arma::mat& mu_mat,
                              const arma::mat& tao_mat, arma::uword p, arma::uword N,
                              arma::mat& Mu_full, arma::mat& Tao_full){
  Mu_full.set_size(N, p);
  Tao_full.set_size(N, p);
  for(arma::uword k = 0; k < p; k++){
    arma::mat Z_block = Z_matrix.rows(k * N, (k + 1) * N - 1);
    Mu_full.col(k)  = Z_block * mu_mat.row(k).t();
    Tao_full.col(k) = Z_block * tao_mat.row(k).t();
  }
}

// accu_log_Tao: sum(log(Tao_full)) over all N*p entries.
arma::vec Metropolis_hastings_portions_cpp(const arma::mat& data_matrix, const arma::mat& Adjacency_matrix_enter, const arma::mat& Causal_effect_matrix_enter, const arma::mat& Mu_full, const arma::mat& Tao_full, double gamma_1, double gamma_result, double accu_log_Tao = NA_REAL){
  const arma::uword p = Causal_effect_matrix_enter.n_cols;

  arma::vec out(3, arma::fill::zeros);

  arma::mat IminusB = arma::eye<arma::mat>(p, p) - Causal_effect_matrix_enter;
  double log_det_IminusB;
  double sign;
  arma::log_det(log_det_IminusB, sign, IminusB);

  // Eps.row(i) == (IminusB * y_i).t() for every data point at once — one
  // BLAS matrix multiply instead of N separate matrix-vector products, and
  // no per-row find()/gather since Mu_full/Tao_full are already precomputed.
  arma::mat Eps  = data_matrix * IminusB.t();        // N x p
  arma::mat diff = Eps - Mu_full;                    // N x p

  // Only the total is used, so accumulate scalars directly instead of
  // materialising two N x 1 temporaries per call.
  const double quad_total = arma::accu((diff % diff) / Tao_full);
  const double log_tau_total =
    R_finite(accu_log_Tao) ? accu_log_Tao : arma::accu(arma::log(Tao_full));

  const double const_term = -0.5 * static_cast<double>(p) * std::log(2.0 * M_PI);
  double first_part = static_cast<double>(Eps.n_rows) * (const_term + log_det_IminusB)
    - 0.5 * (log_tau_total + quad_total);
  out[0] = first_part;

  // Algebraically identical to summing fast_dnorm_log over the nonzero
  // entries, but with no find()/elem() gather and no temporary vector:
  // sum_nz N(b; 0, gamma_1) = -k/2 * log(2*pi*gamma_1) -
  // sum(b^2)/(2*gamma_1) Zero entries contribute exactly 0 to sum(b^2), so
  // accumulating over the whole matrix gives the same number as accumulating
  // over the support.
  const double k_nz = arma::accu(Causal_effect_matrix_enter != 0.0);
  const double sum_sq = arma::accu(arma::square(Causal_effect_matrix_enter));
  out[1] = -0.5 * k_nz * std::log(2.0 * M_PI * gamma_1) - sum_sq / (2.0 * gamma_1);

  // accu() straight off the matrix: no vectorise() copy of p*p doubles.
  const double count1 = arma::accu(Adjacency_matrix_enter != 0.0);
  const double total  = static_cast<double>(Adjacency_matrix_enter.n_elem);
  const double count0 = total - count1;

  // 4) log‐pmf sum for Bernoulli(gamma_result): count1 * log(p) + count0 *
  // log(1-p)
  double lp = std::log(gamma_result);
  double lq = std::log(1.0 - gamma_result);

  double third_part = count1 * lp + count0 * lq;
  out[2] = third_part;

  return out;
}

// R-facing wrapper for Metropolis_hastings_portions_cpp.
// [[Rcpp::export("Metropolis_hastings_portions_cpp")]]
arma::vec Metropolis_hastings_portions_Z_cpp(const arma::mat& data_matrix,
                                             const arma::mat& Adjacency_matrix_enter,
                                             const arma::mat& Causal_effect_matrix_enter,
                                             const arma::mat& Z_matrix_enter,
                                             const arma::mat& mu_mat,
                                             const arma::mat& tao_mat,
                                             double N, double M,
                                             double gamma_1, double gamma_result){
  const arma::uword p  = Causal_effect_matrix_enter.n_cols;
  const arma::uword uN = static_cast<arma::uword>(N);

  if(Z_matrix_enter.n_rows != p * uN)
    Rcpp::stop("Z_matrix must have p * N rows");
  if(mu_mat.n_rows != p || tao_mat.n_rows != p)
    Rcpp::stop("mu_mat and tao_mat must have p rows");
  if(mu_mat.n_cols != static_cast<arma::uword>(M) ||
     tao_mat.n_cols != static_cast<arma::uword>(M))
    Rcpp::stop("mu_mat and tao_mat must have M columns");

  arma::mat Mu_full, Tao_full;
  compute_mixture_mean_var(Z_matrix_enter, mu_mat, tao_mat, p, uN,
                           Mu_full, Tao_full);

  return Metropolis_hastings_portions_cpp(data_matrix, Adjacency_matrix_enter,
                                          Causal_effect_matrix_enter,
                                          Mu_full, Tao_full,
                                          gamma_1, gamma_result);
}

// Exported scoring helper for R-side initialisation code (init_from_seed.R).
// Takes the raw sampler state (Z / mu / tao), builds the per-observation
// mixture mean/variance internally, and returns the three log-posterior
// portions.
// [[Rcpp::export("score_state_v2_cpp")]]
arma::vec score_state_cpp(const arma::mat& data_matrix,
                          const arma::mat& Adjacency_matrix,
                          const arma::mat& Causal_effect_matrix,
                          const arma::mat& Z_matrix,
                          const arma::mat& mu_mat,
                          const arma::mat& tao_mat,
                          double gamma_1, double gamma_result){
  const arma::uword p  = data_matrix.n_cols;
  const arma::uword uN = data_matrix.n_rows;
  arma::mat Mu_full, Tao_full;
  compute_mixture_mean_var(Z_matrix, mu_mat, tao_mat, p, uN, Mu_full, Tao_full);
  return Metropolis_hastings_portions_cpp(
    data_matrix, Adjacency_matrix, Causal_effect_matrix,
    Mu_full, Tao_full, gamma_1, gamma_result);
}

// Progress reporting  Prints at 25%, 50%, 75% and completion. Call once per
// iteration at the END of the loop body, so "25%" means 25% of iterations
// have actually finished rather than started.
inline void report_progress(int i, int total, int& last_q, const char* label){
  if(total <= 0) return;
  const int q = static_cast<int>((4LL * static_cast<long long>(i)) / total);
  if(q <= last_q || q < 1) return;
  last_q = q;
  if(q >= 4)
    Rcpp::Rcout << "[" << label << "] done -- " << total << " iterations\n";
  else
    Rcpp::Rcout << "[" << label << "] " << (q * 25) << "%  ("
                << i << "/" << total << ")\n";
    Rcpp::Rcout.flush();
}

// Per-node Gram cache for the acyclic sampler
namespace nodecache {

struct Cache {
  arma::mat G;        // (p+M) x (p+M) weighted Gram of [data, Z_r]
  arma::vec g;        // (p+M)     U' (y_r / tao)
  double s = 0.0;     // sum(y_r^2 / tao)
  double log_tao = 0.0;   // sum(log(1/sqrt(tao_n)))
  arma::uword p = 0, M = 0;
  mutable std::vector<double> Vs, Ls, gs, ts;   // scratch, grown once

  void build(const arma::mat& data_matrix, const arma::mat& Z_current,
             const arma::vec& tao_list, arma::uword node) {
    p = data_matrix.n_cols; M = Z_current.n_cols;
    const arma::uword n = data_matrix.n_rows, D = p + M;
    arma::vec w = 1.0 / tao_list;
    G.zeros(D, D); g.zeros(D);

    // data block: the only part that costs O(N p^2)
    arma::mat Yw = data_matrix.each_col() % w;
    G.submat(0, 0, p-1, p-1) = Yw.t() * data_matrix;

    // Z is one-hot, so Y'WZ is a per-component column sum (O(N p)) and Z'WZ
    // is diagonal -- no need to touch the N x M block at all.
    const arma::vec y = data_matrix.col(node);
    for(arma::uword r = 0; r < n; ++r){
      arma::uword m = 0;
      for(arma::uword q = 0; q < M; ++q) if(Z_current(r, q) == 1){ m = q; break; }
      const double wr = w(r);
      for(arma::uword c = 0; c < p; ++c){
        const double v = wr * data_matrix(r, c);
        G(c, p + m) += v; G(p + m, c) += v;
      }
      G(p + m, p + m) += wr;
      g(p + m) += wr * y(r);
    }
    for(arma::uword c = 0; c < p; ++c) g(c) = arma::dot(Yw.col(c), y);
    s = arma::accu(w % arma::square(y));
    log_tao = arma::accu(arma::log(1.0 / arma::sqrt(tao_list)));
  }

  // Same formula as log_integral_result_calculator_cpp with need_tao =
  // false.
  double eval(const std::vector<arma::uword>& parents, double N, double gamma_1,
              double b_mu, double which_iter, double num_iter) const {
    const arma::uword k = parents.size(), d = k + M;
    // At this size a LAPACK call costs more than the arithmetic, so V, its
    // Cholesky and the triangular solve use plain arrays allocated once.
    if(Vs.size() < d*d){ Vs.resize(d*d); Ls.resize(d*d); gs.resize(d); ts.resize(d); }
    for(arma::uword b = 0; b < d; ++b){
      const arma::uword cb = (b < k) ? parents[b] : p + (b - k);
      const double* Gcol = G.colptr(cb);
      double* Vcol = &Vs[b*d];
      for(arma::uword a = 0; a < d; ++a)
        Vcol[a] = Gcol[(a < k) ? parents[a] : p + (a - k)];
    }
    for(arma::uword a = 0; a < k; ++a)     Vs[a*d + a]       += 1.0 / gamma_1;
    for(arma::uword a = 0; a < M; ++a)     Vs[(k+a)*d + k+a] += 1.0 / b_mu;
    for(arma::uword a = 0; a < d; ++a)     gs[a] = g((a < k) ? parents[a] : p + (a - k));

    // Cholesky, lower, column-major; then L t = g by forward substitution.
    // quad = g' V^-1 g = ||t||^2, so no inverse is ever formed.
    bool ok = true;
    double logdet = 0.0, quad = 0.0;
    for(arma::uword j = 0; j < d && ok; ++j){
      double sum = Vs[j*d + j];
      for(arma::uword q = 0; q < j; ++q) sum -= Ls[q*d + j] * Ls[q*d + j];
      if(!(sum > 0.0)){ ok = false; break; }
      const double ljj = std::sqrt(sum);
      Ls[j*d + j] = ljj;
      for(arma::uword a = j+1; a < d; ++a){
        double t2 = Vs[j*d + a];
        for(arma::uword q = 0; q < j; ++q) t2 -= Ls[q*d + a] * Ls[q*d + j];
        Ls[j*d + a] = t2 / ljj;
      }
    }
    if(ok){
      for(arma::uword a = 0; a < d; ++a) logdet += std::log(Ls[a*d + a]);
      logdet *= 2.0;
      for(arma::uword a = 0; a < d; ++a){
        double t2 = gs[a];
        for(arma::uword q = 0; q < a; ++q) t2 -= Ls[q*d + a] * ts[q];
        ts[a] = t2 / Ls[a*d + a];
        quad += ts[a] * ts[a];
      }
    } else {
      // not positive definite: fall back to the original pseudo-inverse path
      arma::mat V(&Vs[0], d, d, false, true);
      arma::vec gg(&gs[0], d, false, true);
      double sign; arma::log_det(logdet, sign, V);
      arma::mat V_inv = arma::pinv(V);
      quad = arma::as_scalar(gg.t() * V_inv * gg);
    }

    const double kd = static_cast<double>(k), Md = static_cast<double>(M);
    double first_part = -(N + Md + kd) * std::log(2 * arma::datum::pi)
      + log_tao
    + kd * std::log(std::sqrt(1.0 / gamma_1))
    + Md * std::log(std::sqrt(1.0 / b_mu));
    double first_part_1 = (1 - std::exp(-20 * which_iter / num_iter))
      * (N / 2.0 + Md + kd) * std::log(2 * M_PI);

    double second_part = -0.5 * logdet - 0.5 * s;
    double third_part = (1 + std::exp(-20 * which_iter / num_iter)) * quad;
    return first_part + first_part_1 + second_part + 0.5 * third_part;
  }
};

}  // namespace nodecache

// [[Rcpp::export]]
List BayesSCLingam_cpp(arma::mat data_matrix, double a_mu, double b_mu,
                       double a_gamma, double b_gamma, double a_tao, double b_tao,
                       double a_og_tao, double b_og_tao, double a_gamma_1, double b_gamma_1,
                       double alpha, double M, double num_iter,
                       Rcpp::Nullable<Rcpp::NumericMatrix> init_Adjacency    = R_NilValue,
                       Rcpp::Nullable<Rcpp::NumericMatrix> init_Causal_effect = R_NilValue,
                       Rcpp::Nullable<Rcpp::NumericMatrix> init_mu           = R_NilValue,
                       Rcpp::Nullable<Rcpp::NumericMatrix> init_tao          = R_NilValue,
                       Rcpp::Nullable<Rcpp::NumericMatrix> init_pi           = R_NilValue,
                       Rcpp::Nullable<Rcpp::NumericMatrix> init_Z            = R_NilValue,
                       Rcpp::Nullable<Rcpp::NumericVector> init_gamma_1      = R_NilValue,
                       Rcpp::Nullable<Rcpp::NumericVector> init_gamma_result = R_NilValue){

  const arma::uword p  = static_cast<arma::uword>(data_matrix.n_cols);
  const arma::uword uN = static_cast<arma::uword>(data_matrix.n_rows);
  const arma::uword uM = static_cast<arma::uword>(M);
  const double N       = data_matrix.n_rows;

  // --- Storage
  arma::vec log_likelihood_list(num_iter, arma::fill::zeros);
  arma::vec gamma_1_list(num_iter,        arma::fill::zeros);
  arma::vec gamma_list(num_iter,          arma::fill::zeros);
  arma::mat Adjacency_matrix_list(num_iter,     p * p,  arma::fill::zeros);
  arma::mat Causal_effect_matrix_list(num_iter, p * p,  arma::fill::zeros);
  arma::mat mu_matrix_list(num_iter,  uM * p, arma::fill::zeros);
  arma::mat tao_matrix_list(num_iter, uM * p, arma::fill::zeros);
  arma::mat pi_matrix_list(num_iter,  uM * p, arma::fill::zeros);

  // --- Initialization
  // Each parameter is taken from the corresponding init_* argument when it
  // is supplied (e.g. a DirectLiNGAM warm start built by init_from_seed in
  // R), and falls back to the original prior/random draw when it is NULL.
  auto as_arma = [](Rcpp::NumericMatrix m) -> arma::mat {
    return arma::mat(m.begin(), m.nrow(), m.ncol(), /*copy_aux_mem*/ true);
  };

  arma::mat Adjacency_matrix(p, p, arma::fill::zeros);
  if(init_Adjacency.isNotNull()){
    Adjacency_matrix = as_arma(Rcpp::NumericMatrix(init_Adjacency));
    if(Adjacency_matrix.n_rows != p || Adjacency_matrix.n_cols != p)
      Rcpp::stop("init_Adjacency must be p x p");
    if(!is_dag(Adjacency_matrix))
      Rcpp::stop("init_Adjacency must be a DAG (BayesSCLingam edge moves assume acyclicity)");
  }

  arma::mat Causal_effect_matrix(p, p, arma::fill::zeros);
  if(init_Causal_effect.isNotNull()){
    Causal_effect_matrix = as_arma(Rcpp::NumericMatrix(init_Causal_effect));
    if(Causal_effect_matrix.n_rows != p || Causal_effect_matrix.n_cols != p)
      Rcpp::stop("init_Causal_effect must be p x p");
    // keep coefficients consistent with the (possibly seeded) adjacency
    Causal_effect_matrix %= Adjacency_matrix;
  }

  double gamma_1 = init_gamma_1.isNotNull()
    ? Rcpp::NumericVector(init_gamma_1)(0)
      : rinvgamma_cpp(1, a_gamma, b_gamma)(0);
  double gamma_result = init_gamma_result.isNotNull()
    ? Rcpp::NumericVector(init_gamma_result)(0)
      : rbeta_cpp(1, a_gamma, b_gamma)(0);

  arma::mat mu_mat;
  if(init_mu.isNotNull()){
    mu_mat = as_arma(Rcpp::NumericMatrix(init_mu));
    if(mu_mat.n_rows != p || mu_mat.n_cols != uM)
      Rcpp::stop("init_mu must be p x M");
  } else {
    arma::vec rand_norm_vals_1 = arma::randn<arma::vec>(p * uM);
    arma::vec rand_norm_vals   = a_mu + b_mu * rand_norm_vals_1;
    mu_mat = arma::reshape(rand_norm_vals, uM, p).t();
  }

  arma::mat tao_mat;
  if(init_tao.isNotNull()){
    tao_mat = as_arma(Rcpp::NumericMatrix(init_tao));
    if(tao_mat.n_rows != p || tao_mat.n_cols != uM)
      Rcpp::stop("init_tao must be p x M");
  } else {
    arma::vec rand_inv_gamma_vals = rinvgamma_cpp(p * uM, a_tao, b_tao);
    tao_mat = arma::reshape(rand_inv_gamma_vals, uM, p).t();
  }

  arma::mat pi_mat(p, uM);
  if(init_pi.isNotNull()){
    pi_mat = as_arma(Rcpp::NumericMatrix(init_pi));
    if(pi_mat.n_rows != p || pi_mat.n_cols != uM)
      Rcpp::stop("init_pi must be p x M");
  } else {
    arma::vec alpha_vec(uM, arma::fill::value(alpha));
    for(int i4 = 0; i4 < p; i4++)
      pi_mat.row(i4) = rdirichlet_cpp(alpha_vec).t();
  }

  arma::mat Z_matrix;
  if(init_Z.isNotNull()){
    Z_matrix = as_arma(Rcpp::NumericMatrix(init_Z));
    if(Z_matrix.n_rows != p * uN || Z_matrix.n_cols != uM)
      Rcpp::stop("init_Z must be (p*N) x M");
  } else {
    Z_matrix = arma::zeros<arma::mat>(p * uN, uM);
    for(int i5 = 0; i5 < p; i5++){
      arma::rowvec probs = pi_mat.row(i5);
      for(int j5 = 0; j5 < uN; j5++){
        arma::uword ij = i5 * uN + j5;
        Z_matrix(ij, sample_categorical_cpp(probs)) = 1;
      }
    }
  }

  // Residuals implied by the (possibly seeded) coefficient matrix.
  arma::mat epsilon_mat = ((arma::eye(p, p) - Causal_effect_matrix)
                             * data_matrix.t()).t();

  arma::vec numerator_result(uM,      arma::fill::zeros);
  arma::vec numerator_portion_1(uN,   arma::fill::zeros);
  arma::vec denominator_portion_1(uN, arma::fill::zeros);
  arma::mat practice_prob_mat(uN, uM);
  arma::uvec all_indices = arma::regspace<arma::uvec>(0, p - 1);

  // --- Precompute exclude_j vectors once before the loop
  std::vector<arma::uvec> remaining_index_by_node(p);
  std::vector<arma::uvec> exclude_j_by_node(p);
  for(arma::uword k = 0; k < p; k++){
    remaining_index_by_node[k] = all_indices.elem(arma::find(all_indices != k));
    exclude_j_by_node[k]       = all_indices.elem(arma::find(all_indices != k));
  }

  // --- MH accept lambda
  auto mh_accept = [](double log_r) -> bool {
    if(log_r >= 0) return true;
    return sample_categorical_cpp(
      arma::rowvec{std::exp(log_r), 1.0 - std::exp(log_r)}) == 0;
  };

  int relabel = 1;

  // --- Cached marginal likelihood per node
  // row_val(r) is log_integral_result_calculator_cpp for node r under the
  // CURRENT adjacency row, current tao_mat.row(r) and need_tao = false.
  arma::vec row_val(p);
  std::vector<nodecache::Cache> ncache(p);
  std::vector<double> z_mu(uM), z_inv_tao(uM), z_const(uM), z_lp(uM), z_w(uM);

  // parent list per node, kept in ascending order to match arma::find
  std::vector<std::vector<arma::uword> > parents_of(p);
  auto rebuild_parents = [&](int r){
    parents_of[r].clear();
    for(arma::uword c = 0; c < p; ++c)
      if(Adjacency_matrix(static_cast<arma::uword>(r), c) == 1) parents_of[r].push_back(c);
  };

  // Reachability on the current DAG, allocation-free (stamped visited
  // array). Adding edge a -> b keeps the graph acyclic iff b cannot already
  // reach a; that is the whole is_dag test, without rescanning the p x p
  // matrix.
  std::vector<int> dag_stamp(p, 0); std::vector<int> dag_stack; dag_stack.reserve(p);
  int dag_cur = 0;
  auto can_reach = [&](arma::uword from, arma::uword to){
    if(from == to) return true;
    ++dag_cur; dag_stack.clear(); dag_stack.push_back(static_cast<int>(from));
    dag_stamp[from] = dag_cur;
    while(!dag_stack.empty()){
      const arma::uword v = static_cast<arma::uword>(dag_stack.back()); dag_stack.pop_back();
      // children of v: columns c with Adjacency(c, v) == 1  (c has parent v)
      for(arma::uword c = 0; c < p; ++c){
        if(Adjacency_matrix(c, v) != 1 || dag_stamp[c] == dag_cur) continue;
        if(c == to) return true;
        dag_stamp[c] = dag_cur; dag_stack.push_back(static_cast<int>(c));
      }
    }
    return false;
  };

  // assignment of each observation to a mixture component, per node
  auto tao_list_for = [&](int r, const arma::rowvec& tao_row){
    arma::mat Zc = Z_matrix.rows(static_cast<arma::uword>(r)*static_cast<arma::uword>(N),
                                 (static_cast<arma::uword>(r)+1)*static_cast<arma::uword>(N)-1);
    arma::vec tl(static_cast<arma::uword>(N));
    for(arma::uword n = 0; n < static_cast<arma::uword>(N); ++n){
      arma::uword m = 0; for(arma::uword q = 0; q < Zc.n_cols; ++q) if(Zc(n,q) == 1){ m = q; break; }
      tl(n) = tao_row(m);
    }
    return std::make_pair(Zc, tl);
  };

  // Highest quarter already announced by report_progress (0 = none yet).
  int progress_q = 0;

  // --- Main MCMC loop
  for(int i = 1; i <= num_iter; i++){

    // 1. gamma update
    double total_entries = std::pow(Adjacency_matrix.n_rows, 2);
    double a = a_gamma + arma::accu(Adjacency_matrix);
    double b = b_gamma + total_entries - arma::accu(Adjacency_matrix)
      - Adjacency_matrix.n_rows;
    gamma_result = rbeta_cpp(1, a, b)(0);
    gamma_list(i - 1) = gamma_result;

    // 2. Adjacency matrix + tao update.
    // Rebuild the cache for this iteration
    // (the calculator depends on Z_matrix, tao_mat and the iteration index
    // i, all of which just moved).
    for(int r = 0; r < p; r++){
      auto zt = tao_list_for(r, tao_mat.row(r));
      ncache[r].build(data_matrix, zt.first, zt.second, static_cast<arma::uword>(r));
      rebuild_parents(r);
      row_val(r) = ncache[r].eval(parents_of[r], N, gamma_1, b_mu, i, num_iter);
    }

    for(int i1 = 0; i1 < p; i1++){
      const arma::uvec& remaining_index = remaining_index_by_node[i1];

      arma::vec og_tao = rinvgamma_cpp(uM, a_og_tao, b_og_tao);

      double log_numerator_portion_1 = log_integral_result_calculator_cpp(
        Adjacency_matrix, Z_matrix, data_matrix, i1+1, og_tao,
        true, N, M, gamma_1, b_mu, a_tao, b_tao, i, num_iter);

      // Both sides of the tao ratio go through the original calculator so
      // the comparison stays internally consistent.
      double log_denominator_portion_1 = log_integral_result_calculator_cpp(
        Adjacency_matrix, Z_matrix, data_matrix, i1+1, tao_mat.row(i1).t(),
        true, N, M, gamma_1, b_mu, a_tao, b_tao, i, num_iter);

      if(mh_accept(log_numerator_portion_1 - log_denominator_portion_1)){
        tao_mat.row(i1) = og_tao.t();
        // tao changed for this node: rebuild its Gram cache and cached value
        auto zt = tao_list_for(i1, tao_mat.row(i1));
        ncache[i1].build(data_matrix, zt.first, zt.second, static_cast<arma::uword>(i1));
        row_val(i1) = ncache[i1].eval(parents_of[i1], N, gamma_1, b_mu, i, num_iter);
      }

      for(int j1 = 0; j1 < remaining_index.n_elem; j1++){
        const arma::uword jj = remaining_index(j1);
        const double edge_now = Adjacency_matrix(i1, jj);

        // edge jj -> i1 is safe iff i1 cannot already reach jj
        const bool dag_ok = (edge_now == 1) || !can_reach(i1, jj);
        Adjacency_matrix(i1, jj) = 1;
        if(!dag_ok){
          Adjacency_matrix(i1, jj) = 0;
        } else {
          // value of row i1 WITH the edge; cached when the edge is present
          if(edge_now == 0) rebuild_parents(i1);      // now includes jj
          double log_num_1 = (edge_now == 1) ? row_val(i1)
            : ncache[i1].eval(parents_of[i1], N, gamma_1, b_mu, i, num_iter);

          Adjacency_matrix(i1, jj) = 0;
          // value of row i1 WITHOUT it; cached when the edge is absent
          if(edge_now == 1) rebuild_parents(i1);      // now excludes jj
          double log_den_1 = (edge_now == 0) ? row_val(i1)
            : ncache[i1].eval(parents_of[i1], N, gamma_1, b_mu, i, num_iter);

          double r_log_tmp = log_num_1 - log_den_1
          - (1 + std::exp(-35.0*i/num_iter)) * std::log(1 - gamma_result)
            + (1 + std::exp(-35.0*i/num_iter)) * std::log(gamma_result);

            if(mh_accept(r_log_tmp))
              Adjacency_matrix(i1, jj) = 1;
            rebuild_parents(i1);

            row_val(i1) = (Adjacency_matrix(i1, jj) == 1) ? log_num_1 : log_den_1;

            if(Adjacency_matrix(i1, jj) == 1){
              Adjacency_matrix(i1, jj) = 0;
              // reversed edge i1 -> jj is safe iff jj cannot reach i1 now
              const bool rev_ok = !can_reach(jj, i1);
              Adjacency_matrix(jj, i1) = 1;

              if(!rev_ok){
                Adjacency_matrix(i1, jj) = 1;
                Adjacency_matrix(jj, i1) = 0;
              } else {
                // row i1 without the edge: already computed above. Row i1
                // does not depend on Adjacency(jj, i1), only on its own row.
                double log_numerator_portion_1_1 = log_den_1;
                rebuild_parents(jj);
                double log_numerator_portion_1_2 =
                  ncache[jj].eval(parents_of[jj], N, gamma_1, b_mu, i, num_iter);

                Adjacency_matrix(i1, jj) = 1;
                Adjacency_matrix(jj, i1) = 0;

                double log_numerator_portion_2_1 = log_num_1;
                double log_numerator_portion_2_2 = row_val(jj);

                double r_log_tmp_1 =
                  (log_numerator_portion_1_1 + log_numerator_portion_1_2)
                  - (log_numerator_portion_2_1 + log_numerator_portion_2_2);

                if(mh_accept(r_log_tmp_1)){
                  Adjacency_matrix(i1, jj) = 0;
                  Adjacency_matrix(jj, i1) = 1;
                  row_val(i1) = log_numerator_portion_1_1;
                  row_val(jj) = log_numerator_portion_1_2;
                }
                rebuild_parents(i1); rebuild_parents(jj);
              }
            }
        }
      }
    }
    Adjacency_matrix_list.row(i - 1) = arma::vectorise(Adjacency_matrix).t();

    // 3. mu update
    for(int i2 = 0; i2 < p; i2++){
      arma::mat first_part = Z_matrix.rows(
        i2 * uN, (i2 + 1) * uN - 1);
      first_part.each_row() /= tao_mat.row(i2);

      for(int j2 = 0; j2 < uM; j2++)
        numerator_result(j2) = (a_mu / b_mu)
        + arma::dot(first_part.col(j2), epsilon_mat.col(i2));

      arma::vec denominator_result = (1.0 / b_mu)
        + arma::sum(first_part, 0).t();
      arma::vec mean_vec = numerator_result / denominator_result;
      arma::vec sd_vec   = arma::sqrt(1.0 / denominator_result);
      arma::rowvec mu_row = mean_vec.t()
        + arma::randn<arma::rowvec>(uM) % sd_vec.t();

      if(relabel){
        // ---- full relabeling
        // `ord` is the permutation that puts this node's means in ascending
        // order. Apply it to EVERY component-indexed quantity for node i2 --
        // mu, tao, pi and the Z columns -- so component k keeps its own
        // variance, weight and allocations.
        arma::uvec ord = arma::sort_index(mu_row, "ascend");

        arma::rowvec mu_new(uM), tao_new(uM), pi_new(uM);
        for(arma::uword k = 0; k < uM; k++){
          arma::uword src = ord(k);
          mu_new(k)  = mu_row(src);
          tao_new(k) = tao_mat(i2, src);
          pi_new(k)  = pi_mat(i2, src);
        }
        mu_mat.row(i2)  = mu_new;
        tao_mat.row(i2) = tao_new;
        pi_mat.row(i2)  = pi_new;

        arma::mat Z_block = Z_matrix.rows(i2 * uN, (i2 + 1) * uN - 1);
        Z_matrix.rows(i2 * uN, (i2 + 1) * uN - 1) = Z_block.cols(ord);
      } else {
        // no ordering constraint: labels left wherever the sampler put them
        mu_mat.row(i2) = mu_row;
      }
    }
    mu_matrix_list.row(i - 1) = arma::vectorise(mu_mat).t();

    // 4. Causal effect update
    for(int i3 = 0; i3 < p; i3++){
      arma::mat category_mat = Z_matrix.rows(
        i3 * uN, (i3 + 1) * uN - 1);
      category_mat.each_row() /= tao_mat.row(i3);

      // Hoisted outside j3/z3 loops since it only depends on i3, not j3 or
      // z3 — Causal_effect_matrix(i3,*) doesn't change within this i3
      // iteration until after all j3 are processed
      arma::rowvec mu_row_i3 = mu_mat.row(i3);

      for(int j3 = 0; j3 < p; j3++){
        if(Adjacency_matrix(i3, j3) == 0){
          Causal_effect_matrix(i3, j3) = 0;
        } else {
          const arma::uvec& exclude_j = exclude_j_by_node[j3];

          arma::rowvec ce_row = arma::rowvec(Causal_effect_matrix.row(i3));
          arma::rowvec ce_portion =
            arma::conv_to<arma::rowvec>::from(ce_row.elem(exclude_j));

          for(int z3 = 0; z3 < uN; z3++){
            arma::rowvec Y = data_matrix.row(z3);

            arma::rowvec y_sub =
              arma::conv_to<arma::rowvec>::from(Y.elem(exclude_j));

            double dot_portion     = arma::dot(ce_portion, y_sub);
            arma::rowvec adjusted_mu = Y(i3) - (mu_row_i3 + dot_portion);
            arma::rowvec category_row = Y(j3) * category_mat.row(z3);

            numerator_portion_1(z3)   = arma::dot(category_row, adjusted_mu);
            denominator_portion_1(z3) = arma::accu(
              std::pow(Y(j3), 2) * category_mat.row(z3));
          }

          double numerator_final   = arma::sum(numerator_portion_1);
          double denominator_final = 1.0 / gamma_1
          + arma::sum(denominator_portion_1);
          double mean_result     = numerator_final / denominator_final;
          double variance_result = 1.0 / denominator_final;
          Causal_effect_matrix(i3, j3) = mean_result
          + std::sqrt(variance_result) * arma::randn();
        }
      }
    }
    Causal_effect_matrix_list.row(i - 1) =
      arma::vectorise(Causal_effect_matrix).t();

    // 5. Epsilon update
    epsilon_mat = ((arma::eye(p, p) - Causal_effect_matrix)
                     * data_matrix.t()).t();

    // 6. tao update
    for(int i4 = 0; i4 < p; i4++){
      arma::mat Z_portion = Z_matrix.rows(
        i4 * uN, (i4 + 1) * uN - 1);
      arma::rowvec a_tao_vec = a_tao + arma::sum(Z_portion, 0) / 2.0;
      for(int j4 = 0; j4 < uM; j4++){
        arma::vec eps_col      = epsilon_mat.col(i4);
        double mu_ij           = mu_mat(i4, j4);
        arma::vec squared_diff = arma::square(eps_col - mu_ij);
        arma::vec b_portion    = 0.5 * Z_portion.col(j4) % squared_diff;
        double b_val           = b_tao + arma::sum(b_portion);
        tao_mat(i4, j4)        = rinvgamma_cpp(1, a_tao_vec(j4), b_val)(0);
      }
    }
    tao_matrix_list.row(i - 1) = arma::vectorise(tao_mat).t();

    // 7. gamma_1 update
    double a_1 = a_gamma_1 + arma::accu(Adjacency_matrix) / 2.0;
    double b_1 = b_gamma_1 + arma::accu(
      Adjacency_matrix % (Causal_effect_matrix % Causal_effect_matrix))
      / 2.0;
    gamma_1 = rinvgamma_cpp(1, a_1, b_1)(0);
    gamma_1_list(i - 1) = gamma_1;

    // 8. Z matrix update. Quantities that depend only on the variable are
    // hoisted out of the observation loop. log(pi) and
    // sqrt(tao) depend only on the variable, not the observation, so they
    // move out of the inner loop, and the per-observation rowvec allocations
    // go away.
    for(int i5 = 0; i5 < p; i5++){
      for(int j5 = 0; j5 < uM; j5++){
        const double tao_ij = tao_mat(i5, j5);
        z_mu[j5]      = mu_mat(i5, j5);
        z_inv_tao[j5] = 1.0 / tao_ij;
        z_const[j5]   = std::log(pi_mat(i5, j5))
          - 0.5 * std::log(tao_ij) - 0.5 * std::log(2.0 * M_PI);
      }
      const double* eps_i = epsilon_mat.colptr(i5);
      for(int z5 = 0; z5 < uN; z5++){
        const int iz = i5 * uN + z5;
        const double e = eps_i[z5];
        double max_lp = -std::numeric_limits<double>::infinity();
        for(int j5 = 0; j5 < uM; j5++){
          const double d = e - z_mu[j5];
          z_lp[j5] = z_const[j5] - 0.5 * d * d * z_inv_tao[j5];
          if(z_lp[j5] > max_lp) max_lp = z_lp[j5];
        }
        double total = 0.0;
        for(int j5 = 0; j5 < uM; j5++){ z_w[j5] = std::exp(z_lp[j5] - max_lp); total += z_w[j5]; }
        const double u = arma::randu() * total;
        double cum_prob = 0.0; int sampled = uM - 1;
        for(int m = 0; m < uM; m++){ cum_prob += z_w[m]; if(u < cum_prob){ sampled = m; break; } }
        for(int m = 0; m < uM; m++) Z_matrix(iz, m) = 0.0;
        Z_matrix(iz, sampled) = 1.0;
      }
    }

    // 9. pi update
    for(int i6 = 0; i6 < p; i6++){
      arma::mat Z_portion = Z_matrix.rows(
        i6 * uN, (i6 + 1) * uN - 1);
      arma::rowvec Z_colsum = arma::sum(Z_portion, 0);
      arma::vec alpha_vec_local = Z_colsum.t() + alpha;
      pi_mat.row(i6) = rdirichlet_cpp(alpha_vec_local).t();
    }
    pi_matrix_list.row(i - 1) = arma::vectorise(pi_mat).t();

    // 10. Log likelihood
    arma::mat Mu_full_final, Tao_full_final;
    compute_mixture_mean_var(Z_matrix, mu_mat, tao_mat, p, uN, Mu_full_final, Tao_full_final);
    arma::vec log_parts = Metropolis_hastings_portions_cpp(
      data_matrix, Adjacency_matrix, Causal_effect_matrix,
      Mu_full_final, Tao_full_final, gamma_1, gamma_result);
    log_likelihood_list(i - 1) = log_parts[0];

    report_progress(i, static_cast<int>(num_iter), progress_q, "BayesSCLingam");
  }

  return List::create(
    Named("Adjacency_matrix_list")     = Adjacency_matrix_list,
    Named("Causal_effect_matrix_list") = Causal_effect_matrix_list,
    Named("gamma_list")                = gamma_list,
    Named("gamma_1_list")              = gamma_1_list,
    Named("mu_matrix_list")            = mu_matrix_list,
    Named("tao_matrix_list")           = tao_matrix_list,
    Named("pi_matrix_list")            = pi_matrix_list,
    Named("log_likelihood_list")       = log_likelihood_list
  );
}

// Incremental single-edge scoring.
// Every structure and weight proposal in
// BCD_cpp changes exactly ONE cell of B (and possibly the matching cell of
// Adj).
namespace inc {

struct State {
  arma::mat Eps;              // N x p, = Y * (I-B)^T
  arma::mat Ainv;             // p x p, = (I-B)^{-1}
  double log_abs_det = 0.0;   // log|det(I-B)|
  double quad_total  = 0.0;   // accu((Eps-Mu)^2 / Tao)
  double k_nz        = 0.0;   // number of nonzero entries of B
  double sum_sq      = 0.0;   // accu(B^2)
  double count1      = 0.0;   // number of nonzero entries of Adj
  bool   valid       = false;
};

// Full O(N p^2 + p^3) rebuild. Called once per sampler iteration.
inline bool inc_rebuild(State& S, const arma::mat& Y, const arma::mat& Adj,
                        const arma::mat& B, const arma::mat& Mu,
                        const arma::mat& Tao) {
  const arma::uword p = B.n_cols;
  arma::mat A = arma::eye<arma::mat>(p, p) - B;

  double sign = 0.0;
  arma::log_det(S.log_abs_det, sign, A);
  if(!std::isfinite(S.log_abs_det)) { S.valid = false; return false; }
  if(!arma::inv(S.Ainv, A))         { S.valid = false; return false; }

  S.Eps = Y * A.t();
  const arma::mat d = S.Eps - Mu;
  S.quad_total = arma::accu((d % d) / Tao);
  S.k_nz       = arma::accu(B != 0.0);
  S.sum_sq     = arma::accu(arma::square(B));
  S.count1     = arma::accu(Adj != 0.0);
  S.valid      = true;
  return true;
}

// Tempered log-posterior implied by the state.
inline double inc_logpost(const State& S, double N, double p_d,
                          double accu_log_Tao, double gamma_1,
                          double gamma_result, double lt, double pw) {
  const double loglik =
    N * (-0.5 * p_d * std::log(2.0 * M_PI) + S.log_abs_det)
  - 0.5 * (accu_log_Tao + S.quad_total);
  const double lprior_b = -0.5 * S.k_nz * std::log(2.0 * M_PI * gamma_1)
    - S.sum_sq / (2.0 * gamma_1);
  const double total    = p_d * p_d;
  const double lprior_a = S.count1 * std::log(gamma_result)
    + (total - S.count1) * std::log(1.0 - gamma_result);
  return lt * loglik + pw * (lprior_b + lprior_a);
}

struct Delta {
  bool   ok        = false;
  double d_log_det = 0.0, d_quad = 0.0;
  double d_k = 0.0, d_sumsq = 0.0, d_count1 = 0.0;
  double delta = 0.0, denom = 1.0;
  arma::vec new_eps_col;
  arma::uword r = 0, c = 0;
};

// O(N) score change for setting B(r,c) = b_new and Adj(r,c) = adj_new.
inline Delta inc_edge_delta(const State& S, const arma::mat& Y,
                            const arma::mat& Mu, const arma::mat& Tao,
                            arma::uword r, arma::uword c,
                            double b_old, double b_new,
                            double adj_old, double adj_new) {
  Delta D;
  D.r = r; D.c = c;
  D.delta = b_new - b_old;

  // denom == 0 means (I - B_new) is singular: the proposal is not a valid
  // SEM at all, so it is rejected the same way an unstable one is.
  D.denom = 1.0 - D.delta * S.Ainv(c, r);
  if(!std::isfinite(D.denom) || std::fabs(D.denom) < 1e-10) return D;
  D.d_log_det = std::log(std::fabs(D.denom));

  D.new_eps_col = S.Eps.col(r) - D.delta * Y.col(c);

  const arma::vec dold = S.Eps.col(r)  - Mu.col(r);
  const arma::vec dnew = D.new_eps_col - Mu.col(r);
  D.d_quad = arma::accu((dnew % dnew - dold % dold) / Tao.col(r));

  D.d_k      = (b_new   != 0.0 ? 1.0 : 0.0) - (b_old   != 0.0 ? 1.0 : 0.0);
  D.d_sumsq  = b_new * b_new - b_old * b_old;
  D.d_count1 = (adj_new != 0.0 ? 1.0 : 0.0) - (adj_old != 0.0 ? 1.0 : 0.0);

  D.ok = true;
  return D;
}

inline double inc_delta_logpost(const Delta& D, double N, double gamma_1,
                                double gamma_result, double lt, double pw) {
  const double d_loglik  = N * D.d_log_det - 0.5 * D.d_quad;
  const double d_prior_b = -0.5 * D.d_k * std::log(2.0 * M_PI * gamma_1)
    - D.d_sumsq / (2.0 * gamma_1);
  const double d_prior_a = D.d_count1 * (std::log(gamma_result)
                                           - std::log(1.0 - gamma_result));
  return lt * d_loglik + pw * (d_prior_b + d_prior_a);
}

// Commit an accepted proposal. Ainv columns/rows are copied out first -- the
// update aliases S.Ainv on both sides.
inline void inc_apply(State& S, const Delta& D) {
  S.Eps.col(D.r) = D.new_eps_col;
  S.log_abs_det += D.d_log_det;
  S.quad_total  += D.d_quad;
  S.k_nz        += D.d_k;
  S.sum_sq      += D.d_sumsq;
  S.count1      += D.d_count1;

  const arma::vec    u = S.Ainv.col(D.r);
  const arma::rowvec w = S.Ainv.row(D.c);
  S.Ainv += (D.delta / D.denom) * (u * w);
}

}  // namespace inc

namespace dcs {

// Matrices are stored column-major, exactly like R: element (i, j) of a p x
// p matrix lives at index i + j * p, with 0-based i, j.
inline std::size_t IDX(int i, int j, int p) {
  return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * p;
}

typedef std::vector<std::vector<int> > Graph;

// Adjacency -> successor lists. Successors are in increasing column order,
// matching R's which(Adj[v, ] != 0).
inline Graph build_successors(const std::vector<double>& A, int p) {
  Graph succ(p);
  for (int v = 0; v < p; ++v) {
    for (int w = 0; w < p; ++w) {
      if (A[IDX(v, w, p)] != 0.0) succ[v].push_back(w);
    }
  }
  return succ;
}

// Tarjan's SCC algorithm, iterative. This version keeps an explicit call
// stack instead.
inline std::vector<std::vector<int> > get_sccs(const Graph& succ) {
  const int n = static_cast<int>(succ.size());

  std::vector<int> disc(n, -1), low(n, 0);
  std::vector<char> on_stack(n, 0);
  std::vector<int> stack;
  std::vector<std::vector<int> > sccs;

  int counter = 0;

  std::vector<int> frame_v;        // vertex of each active call
  std::vector<std::size_t> frame_i; // next successor index to visit

  for (int s = 0; s < n; ++s) {
    if (disc[s] != -1) continue;

    disc[s] = low[s] = ++counter;
    stack.push_back(s);
    on_stack[s] = 1;
    frame_v.push_back(s);
    frame_i.push_back(0);

    while (!frame_v.empty()) {
      const int v = frame_v.back();
      const std::size_t i = frame_i.back();

      if (i < succ[v].size()) {
        frame_i.back() = i + 1;
        const int w = succ[v][i];

        if (disc[w] == -1) {
          disc[w] = low[w] = ++counter;
          stack.push_back(w);
          on_stack[w] = 1;
          frame_v.push_back(w);
          frame_i.push_back(0);
        } else if (on_stack[w]) {
          low[v] = std::min(low[v], disc[w]);
        }
      } else {
        // All successors explored: v is finished.
        if (low[v] == disc[v]) {
          std::vector<int> members;
          for (;;) {
            const int u = stack.back();
            stack.pop_back();
            on_stack[u] = 0;
            members.push_back(u);
            if (u == v) break;
          }
          sccs.push_back(members);
        }

        frame_v.pop_back();
        frame_i.pop_back();

        if (!frame_v.empty()) {
          const int parent = frame_v.back();
          low[parent] = std::min(low[parent], low[v]);
        }
      }
    }
  }

  return sccs;
}

// In/out degrees of a node set, counting only edges internal to that set.
inline void scc_degrees(const std::vector<double>& A, int p,
                        const std::vector<int>& members,
                        std::vector<int>& out_deg,
                        std::vector<int>& in_deg) {
  const std::size_t k = members.size();
  out_deg.assign(k, 0);
  in_deg.assign(k, 0);

  for (std::size_t a = 0; a < k; ++a) {
    for (std::size_t b = 0; b < k; ++b) {
      if (A[IDX(members[a], members[b], p)] != 0.0) {
        ++out_deg[a];
        ++in_deg[b];
      }
    }
  }
}

// A digraph's cycles are pairwise vertex-disjoint iff every strongly
// connected component is either a single node (with no self-loop -- the
// caller zeroes the diagonal) or a simple directed cycle, i.e.
inline bool has_disjoint_cycles(const std::vector<double>& A, int p) {
  const std::vector<std::vector<int> > sccs = get_sccs(build_successors(A, p));

  std::vector<int> out_deg, in_deg;

  for (std::size_t s = 0; s < sccs.size(); ++s) {
    const std::vector<int>& members = sccs[s];
    if (members.size() <= 1) continue;

    scc_degrees(A, p, members, out_deg, in_deg);

    for (std::size_t a = 0; a < members.size(); ++a) {
      if (out_deg[a] != 1 || in_deg[a] != 1) return false;
    }
  }

  return true;
}

// Result of the projection.
struct ProjResult {
  std::vector<double> Adj;                      // p x p, column-major
  std::vector<double> B;                        // p x p, column-major
  std::vector<std::pair<int, int> > removed;    // 0-based (row, col) pairs
  bool repaired;
  bool stuck;
  int steps;
  double final_rho;
};

// Spectral radius callback: takes a column-major p x p matrix, returns max
// |eigenvalue|. Injected so the core stays free of any linear algebra
// dependency.
typedef double (*RhoFn)(const std::vector<double>&, int);

// project_to_disjoint_cycle_space()
inline ProjResult project_to_disjoint_cycle_space(std::vector<double> A,
                                                  std::vector<double> B,
                                                  int p,
                                                  int max_steps,
                                                  double stability_target,
                                                  RhoFn rho_fn) {
  ProjResult res;
  res.repaired = false;
  res.stuck = false;
  res.steps = max_steps;
  res.final_rho = std::numeric_limits<double>::quiet_NaN();

  // diag(Adj) <- 0; diag(B) <- 0; Adj <- (Adj != 0) * 1; B[Adj == 0] <- 0
  for (int i = 0; i < p; ++i) {
    A[IDX(i, i, p)] = 0.0;
    B[IDX(i, i, p)] = 0.0;
  }
  for (std::size_t k = 0; k < A.size(); ++k) {
    A[k] = (A[k] != 0.0) ? 1.0 : 0.0;
    if (A[k] == 0.0) B[k] = 0.0;
  }

  std::vector<int> out_deg, in_deg;

  for (int step = 1; step <= max_steps; ++step) {

    if (has_disjoint_cycles(A, p)) {
      for (std::size_t k = 0; k < A.size(); ++k) {
        if (A[k] == 0.0) B[k] = 0.0;
      }

      const double rho = rho_fn(B, p);
      if (std::isfinite(rho) && rho >= stability_target && rho > 0.0) {
        const double scale = stability_target / rho;
        for (std::size_t k = 0; k < B.size(); ++k) B[k] *= scale;
      }

      res.Adj = A;
      res.B = B;
      res.repaired = true;
      res.steps = step - 1;
      res.final_rho = rho_fn(B, p);
      return res;
    }

    const std::vector<std::vector<int> > sccs = get_sccs(build_successors(A, p));

    bool removed_this_step = false;

    for (std::size_t s = 0; s < sccs.size(); ++s) {
      const std::vector<int>& members = sccs[s];
      const std::size_t k = members.size();

      if (k <= 1) continue;

      scc_degrees(A, p, members, out_deg, in_deg);

      // Already a simple directed cycle: leave it alone.
      bool is_cycle = true;
      for (std::size_t a = 0; a < k; ++a) {
        if (out_deg[a] != 1 || in_deg[a] != 1) { is_cycle = false; break; }
      }
      if (is_cycle) continue;

      // Enumerate internal edges in the same order as R's which(sub_adj !=
      // 0, arr.ind = TRUE): column-major over the submatrix indexed by
      // `members` in SCC-stack order. Preserving this order preserves
      // which.min()'s tie-breaking.
      std::vector<int> cand_row, cand_col;
      std::vector<char> is_bad;
      bool any_bad = false;

      for (std::size_t b = 0; b < k; ++b) {        // submatrix column
        for (std::size_t a = 0; a < k; ++a) {      // submatrix row
          if (A[IDX(members[a], members[b], p)] == 0.0) continue;
          cand_row.push_back(members[a]);
          cand_col.push_back(members[b]);
          const bool bad = (out_deg[a] > 1) || (in_deg[b] > 1);
          is_bad.push_back(bad ? 1 : 0);
          if (bad) any_bad = true;
        }
      }

      if (cand_row.empty()) continue;

      // Prefer edges incident to a degree violation.
      int best = -1;
      double best_strength = 0.0;

      for (std::size_t e = 0; e < cand_row.size(); ++e) {
        if (any_bad && !is_bad[e]) continue;

        double strength = std::fabs(B[IDX(cand_row[e], cand_col[e], p)]);
        if (!std::isfinite(strength)) strength = 0.0;  // NA/NaN/Inf -> 0

        if (best < 0 || strength < best_strength) {    // strict: first min wins
          best = static_cast<int>(e);
          best_strength = strength;
        }
      }

      const int r = cand_row[best];
      const int c = cand_col[best];

      A[IDX(r, c, p)] = 0.0;
      B[IDX(r, c, p)] = 0.0;

      res.removed.push_back(std::make_pair(r, c));
      removed_this_step = true;
      break;  // one edge per step, as in the R version
    }

    if (!removed_this_step) {
      res.stuck = true;  // "could not find an internal SCC edge to remove"
      break;
    }
  }

  for (std::size_t k = 0; k < A.size(); ++k) {
    if (A[k] == 0.0) B[k] = 0.0;
  }

  res.Adj = A;
  res.B = B;
  res.repaired = has_disjoint_cycles(A, p);
  return res;
}

}  // namespace dcs

// R interface

static double arma_spectral_radius(const std::vector<double>& M, int p) {
  arma::mat A(const_cast<double*>(&M[0]), p, p, /*copy_aux_mem=*/true);
  arma::cx_vec eigval;
  if (!arma::eig_gen(eigval, A)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return arma::max(arma::abs(eigval));
}

// [[Rcpp::export]]
Rcpp::List project_to_disjoint_cycle_space_cpp(Rcpp::NumericMatrix Adj,
                                               Rcpp::NumericMatrix B,
                                               int max_steps = 10000,
                                               double stability_target = 0.95,
                                               bool verbose = true) {
  const int p = Adj.nrow();

  if (Adj.ncol() != p) Rcpp::stop("Adj must be square.");
  if (B.nrow() != p || B.ncol() != p) Rcpp::stop("B must have the same dimensions as Adj.");
  if (max_steps < 1) Rcpp::stop("max_steps must be >= 1.");

  std::vector<double> Av(Adj.begin(), Adj.end());
  std::vector<double> Bv(B.begin(), B.end());

  // NA propagates strangely through `!= 0` comparisons; treat it as absent.
  for (std::size_t k = 0; k < Av.size(); ++k) {
    if (Rcpp::NumericMatrix::is_na(Av[k])) Av[k] = 0.0;
    if (Rcpp::NumericMatrix::is_na(Bv[k])) Bv[k] = 0.0;
  }

  dcs::ProjResult res = dcs::project_to_disjoint_cycle_space(
    Av, Bv, p, max_steps, stability_target, &arma_spectral_radius);

  if (res.stuck) {
    Rcpp::warning("Projection got stuck: could not find an internal SCC edge to remove.");
  }

  Rcpp::NumericMatrix Adj_out(p, p);
  Rcpp::NumericMatrix B_out(p, p);
  std::copy(res.Adj.begin(), res.Adj.end(), Adj_out.begin());
  std::copy(res.B.begin(), res.B.end(), B_out.begin());

  // Carry dimnames through, if the input had any.
  Rcpp::RObject dn = Adj.attr("dimnames");
  if (!dn.isNULL()) {
    Adj_out.attr("dimnames") = dn;
    B_out.attr("dimnames") = dn;
  }

  const int n_removed = static_cast<int>(res.removed.size());
  Rcpp::IntegerMatrix removed(n_removed, 2);
  for (int e = 0; e < n_removed; ++e) {
    removed(e, 0) = res.removed[e].first + 1;   // back to 1-based for R
    removed(e, 1) = res.removed[e].second + 1;
  }
  removed.attr("dimnames") = Rcpp::List::create(
    R_NilValue,
    Rcpp::CharacterVector::create("row_child", "col_parent"));

  if (verbose && res.repaired && !res.stuck) {
    Rcpp::Rcout << "Projection successful.\n";
    Rcpp::Rcout << "Steps: " << res.steps << "\n";
    Rcpp::Rcout << "Edges removed: " << n_removed << "\n";
    Rcpp::Rcout << "Final disjoint: TRUE\n";
    Rcpp::Rcout << "Final rho: " << arma_spectral_radius(res.B, p) << "\n";
  }

  return Rcpp::List::create(
    Rcpp::Named("AdjacencyMatrix") = Adj_out,
    Rcpp::Named("CausalEffectMatrix") = B_out,
    Rcpp::Named("removed_edges") = removed,
    Rcpp::Named("repaired") = res.repaired,
    Rcpp::Named("steps") = res.steps);
}

// [[Rcpp::export]]
bool has_disjoint_cycles_cpp(Rcpp::NumericMatrix Adj) {
  const int p = Adj.nrow();
  if (Adj.ncol() != p) Rcpp::stop("Adj must be square.");

  std::vector<double> Av(Adj.begin(), Adj.end());
  for (std::size_t k = 0; k < Av.size(); ++k) {
    if (Rcpp::NumericMatrix::is_na(Av[k])) Av[k] = 0.0;
  }
  for (int i = 0; i < p; ++i) Av[dcs::IDX(i, i, p)] = 0.0;

  return dcs::has_disjoint_cycles(Av, p);
}

// [[Rcpp::export]]
Rcpp::List get_sccs_cpp(Rcpp::NumericMatrix Adj) {
  const int p = Adj.nrow();
  if (Adj.ncol() != p) Rcpp::stop("Adj must be square.");

  std::vector<double> Av(Adj.begin(), Adj.end());
  for (std::size_t k = 0; k < Av.size(); ++k) {
    if (Rcpp::NumericMatrix::is_na(Av[k])) Av[k] = 0.0;
  }

  const std::vector<std::vector<int> > sccs =
    dcs::get_sccs(dcs::build_successors(Av, p));

  Rcpp::List out(sccs.size());
  for (std::size_t s = 0; s < sccs.size(); ++s) {
    Rcpp::IntegerVector v(sccs[s].size());
    for (std::size_t a = 0; a < sccs[s].size(); ++a) v[a] = sccs[s][a] + 1;
    out[s] = v;
  }
  return out;
}

// Two-phase cyclic causal discovery sampler: an unconstrained phase, then a
// phase restricted to graphs whose cycles are vertex-disjoint. a_og_tao / b_og_tao are
// deliberately NOT parameters here.
namespace fastchk {

// ---- Stability: is the spectral radius of B below 1?
// A proposal changes one cell, so row and column sums of |B| are cached and
// only the affected row and column are re-summed. Cheap norm bounds settle
// most cases; otherwise eigenvalues are computed per strongly connected
// block, since B is block triangular in SCC order.
struct Stability {
  arma::uword p = 0;
  std::vector<double> row_abs, col_abs;
  long dirty_r = -1, dirty_c = -1;

  // True when the CURRENT (last accepted) B is known to be stable. Set by
  // refresh_all(); every accepted move passes check(), so it stays true.
  bool state_stable = false;

  // scratch, allocated once
  std::vector<int>  t_idx, t_low, t_stack, t_members, members;
  std::vector<char> t_on;
  std::vector<int>  fwd, bwd, inset;
  int stamp = 0;
  std::vector<int>  bfs;
  std::vector<std::pair<int, arma::uword> > t_work;
  std::vector<double> brow, bcol;

  void init(arma::uword p_) {
    p = p_;
    row_abs.assign(p, 0.0);  col_abs.assign(p, 0.0);
    t_idx.assign(p, -1);     t_low.assign(p, 0);   t_on.assign(p, 0);
    fwd.assign(p, 0); bwd.assign(p, 0); inset.assign(p, 0); stamp = 0;
    t_stack.reserve(p); t_members.reserve(p); members.reserve(p);
    t_work.reserve(p);  bfs.reserve(p); brow.resize(p); bcol.resize(p);
    dirty_r = dirty_c = -1; state_stable = false;
  }

  void refresh_row(const arma::mat& C, arma::uword r) {
    double s = 0.0;
    for(arma::uword j = 0; j < p; ++j) s += std::fabs(C(r, j));
    row_abs[r] = s;
  }
  void refresh_col(const arma::mat& C, arma::uword c) {
    const double* x = C.colptr(c);
    double s = 0.0;
    for(arma::uword k = 0; k < p; ++k) s += std::fabs(x[k]);
    col_abs[c] = s;
  }
  // Call whenever B changed outside a proposal (once per iteration).
  void refresh_all(const arma::mat& C) {
    for(arma::uword r = 0; r < p; ++r) refresh_row(C, r);
    for(arma::uword c = 0; c < p; ++c) refresh_col(C, c);
    dirty_r = dirty_c = -1;
    state_stable = global_bounds_ok() || blocks_ok(C, false);
  }

  bool global_bounds_ok() const {
    double mr = 0.0, mc = 0.0;
    for(arma::uword k = 0; k < p; ++k) {
      if(row_abs[k] > mr) mr = row_abs[k];
      if(col_abs[k] > mc) mc = col_abs[k];
    }
    return (mr < 1.0) || (mc < 1.0) || (mr * mc < 1.0);
  }

  // Spectral radius of the principal submatrix on `members` below 1? Norm
  // bounds on the block first (they ignore every entry outside it, which is
  // why they succeed where whole-matrix bounds fail), then an eigen-
  // decomposition of the block only if they cannot decide.
  bool block_ok(const arma::mat& C, const std::vector<int>& m) {
    const arma::uword k = m.size();
    if(k == 1) {
      const arma::uword v = static_cast<arma::uword>(m[0]);
      return std::fabs(C(v, v)) < 1.0;
    }
    double mr = 0.0, mc = 0.0;
    for(arma::uword a = 0; a < k; ++a) { brow[a] = 0.0; bcol[a] = 0.0; }
    for(arma::uword b = 0; b < k; ++b)
      for(arma::uword a = 0; a < k; ++a) {
        const double x = std::fabs(C(static_cast<arma::uword>(m[a]),
                                     static_cast<arma::uword>(m[b])));
        brow[a] += x; bcol[b] += x;
      }
    for(arma::uword a = 0; a < k; ++a) {
      if(brow[a] > mr) mr = brow[a];
      if(bcol[a] > mc) mc = bcol[a];
    }
    if(mr < 1.0 || mc < 1.0 || mr * mc < 1.0) return true;
    arma::mat sub(k, k);
    for(arma::uword b = 0; b < k; ++b)
      for(arma::uword a = 0; a < k; ++a)
        sub(a, b) = C(static_cast<arma::uword>(m[a]), static_cast<arma::uword>(m[b]));
    arma::cx_vec ev = arma::eig_gen(sub);
    return arma::abs(ev).max() < 1.0;
  }

  // Tarjan over the whole graph (restricted = false) or over the vertices
  // marked inset[v] == stamp (restricted = true); every SCC must pass
  // block_ok. Edge v -> w wherever C(v,w) != 0.
  bool blocks_ok(const arma::mat& C, bool restricted) {
    for(arma::uword v = 0; v < p; ++v) { t_idx[v] = -1; t_on[v] = 0; }
    t_stack.clear();
    int counter = 0;
    for(arma::uword s = 0; s < p; ++s) {
      if(t_idx[s] != -1) continue;
      if(restricted && inset[s] != stamp) continue;
      t_work.clear();
      t_work.push_back(std::make_pair(static_cast<int>(s), arma::uword(0)));
      t_idx[s] = t_low[s] = counter++;
      t_stack.push_back(static_cast<int>(s)); t_on[s] = 1;
      while(!t_work.empty()) {
        const int v = t_work.back().first;
        arma::uword w = t_work.back().second;
        bool descended = false;
        for(; w < p; ++w) {
          if(w == static_cast<arma::uword>(v)) continue;
          if(restricted && inset[w] != stamp) continue;
          if(C(static_cast<arma::uword>(v), w) == 0.0) continue;
          if(t_idx[w] == -1) {
            t_work.back().second = w + 1;
            t_idx[w] = t_low[w] = counter++;
            t_stack.push_back(static_cast<int>(w)); t_on[w] = 1;
            t_work.push_back(std::make_pair(static_cast<int>(w), arma::uword(0)));
            descended = true; break;
          } else if(t_on[w]) {
            if(t_idx[w] < t_low[v]) t_low[v] = t_idx[w];
          }
        }
        if(descended) continue;
        if(t_low[v] == t_idx[v]) {
          t_members.clear();
          int x;
          do { x = t_stack.back(); t_stack.pop_back(); t_on[x] = 0; t_members.push_back(x); }
          while(x != v);
          if(!block_ok(C, t_members)) return false;
        }
        t_work.pop_back();
        if(!t_work.empty()) {
          const int u = t_work.back().first;
          if(t_low[v] < t_low[u]) t_low[u] = t_low[v];
        }
      }
    }
    return true;
  }

  // SCC of r: {v : r ~> v} intersect {v : v ~> r}, collected in `members`
  // and marked inset[v] == stamp.
  void scc_of(const arma::mat& C, arma::uword r, bool virt, arma::uword c) {
    ++stamp;
    bfs.clear(); bfs.push_back(static_cast<int>(r)); fwd[r] = stamp;
    while(!bfs.empty()) {
      const arma::uword v = static_cast<arma::uword>(bfs.back()); bfs.pop_back();
      for(arma::uword w = 0; w < p; ++w) {
        if(w == v || fwd[w] == stamp) continue;
        if(C(v, w) != 0.0 || (virt && v == r && w == c)) {
          fwd[w] = stamp; bfs.push_back(static_cast<int>(w));
        }
      }
    }
    members.clear();
    bfs.clear(); bfs.push_back(static_cast<int>(r)); bwd[r] = stamp;
    while(!bfs.empty()) {
      const arma::uword v = static_cast<arma::uword>(bfs.back()); bfs.pop_back();
      if(fwd[v] == stamp) { members.push_back(static_cast<int>(v)); inset[v] = stamp; }
      const double* col = C.colptr(v);
      for(arma::uword w = 0; w < p; ++w) {
        if(w == v || bwd[w] == stamp) continue;
        if(col[w] != 0.0 || (virt && v == c && w == r)) {
          bwd[w] = stamp; bfs.push_back(static_cast<int>(w));
        }
      }
    }
  }

  // Call after the candidate's cell (r,c) has been written into C; old_val
  // is what the cell held before.
  bool check(const arma::mat& C, arma::uword r, arma::uword c, double old_val) {
    if(dirty_r >= 0 && static_cast<arma::uword>(dirty_r) != r)
      refresh_row(C, static_cast<arma::uword>(dirty_r));
    if(dirty_c >= 0 && static_cast<arma::uword>(dirty_c) != c)
      refresh_col(C, static_cast<arma::uword>(dirty_c));
    refresh_row(C, r); refresh_col(C, c);
    dirty_r = static_cast<long>(r); dirty_c = static_cast<long>(c);

    if(global_bounds_ok()) return true;           // the original cheap test
    if(!state_stable) return blocks_ok(C, false);  // no shortcut: check all blocks

    // Current state is stable, and B is block triangular in SCC order with
    // eigenvalues = union of the diagonal blocks'. Only blocks whose entries
    // changed can have new eigenvalues.
    const double new_val = C(r, c);
    if(new_val != 0.0) {
      // weight change (partition unchanged) or addition (may merge SCCs; the
      // merged SCC is the one containing r). Entry is inside a diagonal
      // block iff r and c share an SCC in the new graph.
      scc_of(C, r, false, c);
      if(inset[c] != stamp) return true;           // off-block: nothing changed
      return block_ok(C, members);
    }
    // removal: if r and c shared an SCC K before, K may split into pieces
    // whose own eigenvalues were never examined, so re-check inside K.
    scc_of(C, r, true, c);                          // K = old SCC of r
    if(inset[c] != stamp) return true;              // entry was off-block
    return blocks_ok(C, true);                      // Tarjan restricted to K
  }
};

// ---- Disjoint cycles: the one case the shortcuts cannot settle
// Reached only when an ADDITION r -> c closes a cycle (c already reached r)
// and the graph before the addition satisfied the property.
struct Topology {
  arma::uword p = 0;
  std::vector<int> fwd, bwd, stack, members;
  int cur = 0;

  void init(arma::uword p_) {
    p = p_;
    fwd.assign(p, 0); bwd.assign(p, 0);
    stack.reserve(p); members.reserve(p);
    cur = 0;
  }

  // Adj(r,c) must already be 1.
  bool closed_cycle_is_simple(const arma::mat& A, arma::uword r) {
    ++cur;
    // forward from r: successors of v are w with A(v,w) != 0
    stack.clear(); stack.push_back(static_cast<int>(r)); fwd[r] = cur;
    while(!stack.empty()) {
      const arma::uword v = static_cast<arma::uword>(stack.back()); stack.pop_back();
      for(arma::uword w = 0; w < p; ++w) {
        if(w == v || A(v, w) == 0.0 || fwd[w] == cur) continue;
        fwd[w] = cur; stack.push_back(static_cast<int>(w));
      }
    }
    // backward from r: predecessors of v are w with A(w,v) != 0
    members.clear();
    stack.clear(); stack.push_back(static_cast<int>(r)); bwd[r] = cur;
    while(!stack.empty()) {
      const arma::uword v = static_cast<arma::uword>(stack.back()); stack.pop_back();
      if(fwd[v] == cur) members.push_back(static_cast<int>(v));
      const double* col = A.colptr(v);
      for(arma::uword w = 0; w < p; ++w) {
        if(w == v || col[w] == 0.0 || bwd[w] == cur) continue;
        bwd[w] = cur; stack.push_back(static_cast<int>(w));
      }
    }
    // members = SCC of r; each needs exactly one in-SCC successor and
    // predecessor
    for(int vi : members) {
      const arma::uword v = static_cast<arma::uword>(vi);
      int outd = 0, ind = 0;
      for(int wi : members) {
        const arma::uword w = static_cast<arma::uword>(wi);
        if(w == v) continue;
        if(A(v, w) != 0.0) ++outd;
        if(A(w, v) != 0.0) ++ind;
      }
      if(outd != 1 || ind != 1) return false;
    }
    return true;
  }
};

}  // namespace fastchk

// [[Rcpp::export("BCD_v2_two_phase_cpp")]]
List BCD_cpp(arma::mat data_matrix, double a_mu, double b_mu,
             double a_gamma, double b_gamma, double a_tao, double b_tao,
             double a_gamma_1, double b_gamma_1,
             double alpha, double M, double num_iter,
             double burn_in_iterations = 70000,
             Rcpp::Nullable<Rcpp::NumericMatrix> init_Adjacency    = R_NilValue,
             Rcpp::Nullable<Rcpp::NumericMatrix> init_Causal_effect = R_NilValue,
             Rcpp::Nullable<Rcpp::NumericMatrix> init_mu           = R_NilValue,
             Rcpp::Nullable<Rcpp::NumericMatrix> init_tao          = R_NilValue,
             Rcpp::Nullable<Rcpp::NumericMatrix> init_pi           = R_NilValue,
             Rcpp::Nullable<Rcpp::NumericMatrix> init_Z            = R_NilValue,
             Rcpp::Nullable<Rcpp::NumericVector> init_gamma_1      = R_NilValue,
             Rcpp::Nullable<Rcpp::NumericVector> init_gamma_result = R_NilValue){

  const arma::uword p  = static_cast<arma::uword>(data_matrix.n_cols);
  const arma::uword uN = static_cast<arma::uword>(data_matrix.n_rows);
  const arma::uword uM = static_cast<arma::uword>(M);
  const double N       = data_matrix.n_rows;

  // --- Phase schedule (needed here to size the storage)
  // Iterations 1..burn_in are unconstrained; constrained_start onward are
  // restricted to disjoint cycles, and constrained_start..anneal_end-1 are
  // tempered.
  const int n_iter_i           = static_cast<int>(num_iter);
  const int burn_in_iters_i    = static_cast<int>(burn_in_iterations);
  const int constrained_start  = burn_in_iters_i + 1;
  // std::max guards num_iter <= burn_in_iterations (no constrained phase)
  const int anneal_len = std::max(0, static_cast<int>(
    std::floor(0.4 * (n_iter_i - burn_in_iters_i))));
  const int anneal_end = constrained_start + anneal_len;

  // --- Storage: only iterations i >= anneal_end
  // Earlier iterations are not draws from the target posterior: in the
  // unconstrained phase gamma_result is deterministically ramped, and in the
  // annealing window the accept ratio uses likelihood_temp != 1 and
  // prior_weight != 1.
  const int n_keep = std::max(0, n_iter_i - anneal_end + 1);

  arma::vec gamma_1_list(n_keep, arma::fill::zeros);
  arma::vec gamma_list(n_keep, arma::fill::zeros);
  arma::mat Adjacency_matrix_list(n_keep, p * p, arma::fill::zeros);
  arma::mat Causal_effect_matrix_list(n_keep, p * p, arma::fill::zeros);
  arma::mat mu_matrix_list(n_keep,  uM * p, arma::fill::zeros);
  arma::mat tao_matrix_list(n_keep, uM * p, arma::fill::zeros);
  arma::mat pi_matrix_list(n_keep,  uM * p, arma::fill::zeros);

  if(n_keep == 0)
    Rcpp::warning("No post-annealing iterations: num_iter is too small "
                    "relative to burn_in_iterations, so every trace is empty.");

  // --- Initialization
  // Each parameter is taken from the corresponding init_* argument when it
  // is supplied (e.g. a DirectLiNGAM warm start built by init_from_seed in
  // R), and falls back to the original prior/random draw when it is NULL.
  auto as_arma = [](Rcpp::NumericMatrix m) -> arma::mat {
    return arma::mat(m.begin(), m.nrow(), m.ncol(), /*copy_aux_mem*/ true);
  };

  arma::mat Adjacency_matrix(p, p, arma::fill::zeros);
  if(init_Adjacency.isNotNull()){
    Adjacency_matrix = as_arma(Rcpp::NumericMatrix(init_Adjacency));
    if(Adjacency_matrix.n_rows != p || Adjacency_matrix.n_cols != p)
      Rcpp::stop("init_Adjacency must be p x p");
    // NOTE: the DAG check from BayesSCLingam_cpp is deliberately NOT applied
    // here. This is the cyclic sampler -- the unconstrained phase is meant
    // to visit graphs with feedback loops, so a cyclic warm start is legal.
    Adjacency_matrix.diag().zeros();
  }

  arma::mat Causal_effect_matrix(p, p, arma::fill::zeros);
  if(init_Causal_effect.isNotNull()){
    Causal_effect_matrix = as_arma(Rcpp::NumericMatrix(init_Causal_effect));
    if(Causal_effect_matrix.n_rows != p || Causal_effect_matrix.n_cols != p)
      Rcpp::stop("init_Causal_effect must be p x p");
    // keep coefficients consistent with the (possibly seeded) adjacency
    Causal_effect_matrix %= Adjacency_matrix;
  }

  double gamma_1 = init_gamma_1.isNotNull()
    ? Rcpp::NumericVector(init_gamma_1)(0)
      : rinvgamma_cpp(1, a_gamma, b_gamma)(0);
  double gamma_result = init_gamma_result.isNotNull()
    ? Rcpp::NumericVector(init_gamma_result)(0)
      : rbeta_cpp(1, a_gamma, b_gamma)(0);

  arma::mat mu_mat;
  if(init_mu.isNotNull()){
    mu_mat = as_arma(Rcpp::NumericMatrix(init_mu));
    if(mu_mat.n_rows != p || mu_mat.n_cols != uM)
      Rcpp::stop("init_mu must be p x M");
  } else {
    arma::vec rand_norm_vals_1 = arma::randn<arma::vec>(p * uM);
    arma::vec rand_norm_vals   = a_mu + b_mu * rand_norm_vals_1;
    mu_mat = arma::reshape(rand_norm_vals, uM, p).t();
  }

  arma::mat tao_mat;
  if(init_tao.isNotNull()){
    tao_mat = as_arma(Rcpp::NumericMatrix(init_tao));
    if(tao_mat.n_rows != p || tao_mat.n_cols != uM)
      Rcpp::stop("init_tao must be p x M");
  } else {
    arma::vec rand_inv_gamma_vals = rinvgamma_cpp(p * uM, a_tao, b_tao);
    tao_mat = arma::reshape(rand_inv_gamma_vals, uM, p).t();
  }

  arma::mat pi_mat(p, uM);
  if(init_pi.isNotNull()){
    pi_mat = as_arma(Rcpp::NumericMatrix(init_pi));
    if(pi_mat.n_rows != p || pi_mat.n_cols != uM)
      Rcpp::stop("init_pi must be p x M");
  } else {
    arma::vec alpha_vec(uM, arma::fill::value(alpha));
    for(int i4 = 0; i4 < p; i4++)
      pi_mat.row(i4) = rdirichlet_cpp(alpha_vec).t();
  }

  arma::mat Z_matrix;
  if(init_Z.isNotNull()){
    Z_matrix = as_arma(Rcpp::NumericMatrix(init_Z));
    if(Z_matrix.n_rows != p * uN || Z_matrix.n_cols != uM)
      Rcpp::stop("init_Z must be (p*N) x M");
  } else {
    Z_matrix = arma::zeros<arma::mat>(p * uN, uM);
    for(int i5 = 0; i5 < p; i5++){
      arma::rowvec probs = pi_mat.row(i5);
      for(int j5 = 0; j5 < uN; j5++){
        arma::uword ij = i5 * uN + j5;
        Z_matrix(ij, sample_categorical_cpp(probs)) = 1;
      }
    }
  }

  // Residuals implied by the (possibly seeded) coefficient matrix.
  arma::mat epsilon_mat = ((arma::eye(p, p) - Causal_effect_matrix)
                             * data_matrix.t()).t();

  arma::vec numerator_result(uM,      arma::fill::zeros);
  arma::vec numerator_portion_1(uN,   arma::fill::zeros);
  arma::vec denominator_portion_1(uN, arma::fill::zeros);
  arma::mat practice_prob_mat(uN, uM);

  // Per-component scratch for the Z update (step 8). Allocated once for the
  // whole run rather than twice per observation per iteration.
  std::vector<double> z_mu(static_cast<std::size_t>(uM));
  std::vector<double> z_inv_tao(static_cast<std::size_t>(uM));
  std::vector<double> z_const(static_cast<std::size_t>(uM));
  std::vector<double> z_lp(static_cast<std::size_t>(uM));
  std::vector<double> z_w(static_cast<std::size_t>(uM));
  // --- Candidate-column index sets, built once before the sampler loop
  // remaining_indices[k] is every node index except k, i.e. the columns that
  // are legal edge targets for row k (the diagonal is not a representable
  // edge).
  arma::uvec all_indices = arma::regspace<arma::uvec>(0, p - 1);

  std::vector<arma::uvec> remaining_indices(p);
  for(arma::uword k = 0; k < p; k++){
    remaining_indices[k] = all_indices.elem(arma::find(all_indices != k));
  }

  // --- MH accept lambda
  auto mh_accept = [](double log_r) -> bool {
    if(log_r >= 0) return true;
    return sample_categorical_cpp(
      arma::rowvec{std::exp(log_r), 1.0 - std::exp(log_r)}) == 0;
  };

  int relabel = 1;

  // --- Stability check helper
  // Returns true if the matrix is stable (max eigenvalue modulus < 1) Uses a
  // cheap row-sum pre-filter before falling through to full eig_gen.

  // --- Disjoint-cycle check helper
  // Returns true if every nontrivial strongly connected component (SCC) of
  // the directed graph implied by Adj is itself a single simple cycle, i.e.
  // no two feedback loops share a vertex.
  auto has_disjoint_cycles = [](const arma::mat& Adj) -> bool {
    // Delegates to dcs::has_disjoint_cycles (iterative Tarjan) instead of
    // keeping a second, recursive copy of the same algorithm in this scope.
    // arma::mat is column-major, the same layout dcs:: expects, so this is a
    // straight linear read.
    const int pp = static_cast<int>(Adj.n_rows);
    std::vector<double> A(Adj.begin(), Adj.end());
    for(int d = 0; d < pp; d++) A[dcs::IDX(d, d, pp)] = 0.0;
    return dcs::has_disjoint_cycles(A, pp);
  };

  // --- Reachability, allocation-free
  // Buffers are hoisted out of the candidate loop and a monotonically
  // increasing stamp replaces clearing the visited array, so a query costs
  // O(V+E) with zero heap traffic.
  std::vector<int> reach_stamp(static_cast<std::size_t>(p), 0);
  std::vector<int> reach_stack;
  reach_stack.reserve(static_cast<std::size_t>(p));
  int reach_cur = 0;

  auto reaches = [&](arma::uword from, arma::uword to) -> bool {
    if(from == to) return true;
    ++reach_cur;
    reach_stack.clear();
    reach_stack.push_back(static_cast<int>(from));
    reach_stamp[from] = reach_cur;
    while(!reach_stack.empty()){
      const int v = reach_stack.back();
      reach_stack.pop_back();
      for(arma::uword w = 0; w < p; w++){
        if(Adjacency_matrix(static_cast<arma::uword>(v), w) == 0.0) continue;
        if(w == to) return true;
        if(reach_stamp[w] != reach_cur){
          reach_stamp[w] = reach_cur;
          reach_stack.push_back(static_cast<int>(w));
        }
      }
    }
    return false;
  };

  // --- Disjoint-cycle check with two exact shortcuts
  // Call AFTER toggling cell (r,c); old_adj is what that cell held before.
  // Both shortcuts are exact, not heuristics:  1.
  fastchk::Stability stab;      stab.init(p);
  fastchk::Topology  cycle_chk; cycle_chk.init(p);

  // Every shortcut below relies on the graph BEFORE the toggle already
  // satisfying the disjoint-cycle property. That holds after a successful
  // projection and is preserved by every accepted move, but it is checked
  // once per iteration anyway: if it ever fails (e.g.
  bool topo_valid = true;

  auto disjoint_cycles_ok = [&](arma::uword r, arma::uword c,
                                double old_adj) -> bool {
                                  if(!topo_valid) return has_disjoint_cycles(Adjacency_matrix);

                                  if(old_adj != 0.0) return true;                  // shortcut 1: removal

                                  Adjacency_matrix(r, c) = 0.0;                    // look at G without e
                                  const bool closes_cycle = reaches(c, r);
                                  Adjacency_matrix(r, c) = 1.0;
                                  if(!closes_cycle) return true;                   // shortcut 2

                                  return cycle_chk.closed_cycle_is_simple(Adjacency_matrix, r);
                                };

  // --- Projection helper
  // Snaps the current state into the disjoint-cycle space: drops the weakest
  // edge of each offending SCC until every SCC is a simple cycle, then
  // rescales B to spectral radius 0.95. Used exactly once, at the
  // unconstrained -> constrained handoff.
  auto project_state = [&](arma::mat& Adj, arma::mat& B) -> void {
    const int pp = static_cast<int>(Adj.n_rows);
    std::vector<double> Av(Adj.begin(), Adj.end());
    std::vector<double> Bv(B.begin(),   B.end());

    dcs::ProjResult pr = dcs::project_to_disjoint_cycle_space(
      Av, Bv, pp, /*max_steps=*/10000, /*stability_target=*/0.95,
      &arma_spectral_radius);

    std::copy(pr.Adj.begin(), pr.Adj.end(), Adj.begin());
    std::copy(pr.B.begin(),   pr.B.end(),   B.begin());

    if(!pr.repaired)
      Rcpp::warning("Projection at the constrained handoff did not reach a "
                      "disjoint-cycle graph; the constrained phase may reject "
                      "every proposal.");
  };

  // --- Mixture mean/variance lookup, precomputed once
  // Only depends on Z_matrix/mu_mat/tao_mat, which stay fixed across the
  // O(p^2) structure and weight proposals within an iteration — so this is
  // computed once per iteration (after Z/mu/tao actually change) instead of
  // being redone inside every single Metropolis_hastings_portions_cpp call.
  arma::mat Mu_full, Tao_full;
  // Cached sum(log(Tao_full)); refreshed wherever Tao_full is rebuilt.
  double accu_log_Tao = NA_REAL;

  // Incremental single-edge scoring state (see namespace inc above). Rebuilt
  // once per iteration, then updated in O(N) per accepted move.
  inc::State S;
  compute_mixture_mean_var(Z_matrix, mu_mat, tao_mat, p, uN, Mu_full, Tao_full);
  accu_log_Tao = arma::accu(arma::log(Tao_full));

  // Tempered log-posterior of the current state. Set unconditionally by
  // inc_logpost at the top of every iteration, so the initial value is never
  // read -- no need to score the starting state here.
  double log_current = 0.0;

  // --- Phase schedule
  // Iterations 1 .. burn_in_iterations       : UNCONSTRAINED (cycles may
  // overlap) Iterations constrained_start ..

  // Annealing weights. likelihood_temp flattens the likelihood and
  // prior_weight sharpens the priors during the annealing window; both are
  // passed straight into inc_logpost / inc_delta_logpost, which apply them
  // as lt * loglik + pw * (prior_b + prior_a).
  double likelihood_temp = 1.0;
  double prior_weight    = 1.0;

  if(burn_in_iters_i >= n_iter_i)
    Rcpp::warning("burn_in_iterations >= num_iter: the constrained "
                    "(disjoint-cycle) phase will never run.");

  bool projected          = false;
  const double burn_in    = 1000;   // proposal-sd warmup (separate schedule)

  // Highest quarter already announced by report_progress (0 = none yet).
  int progress_q = 0;

  for(int i = 1; i <= n_iter_i; i++){

    // Stored only from anneal_end on; row k <-> iteration anneal_end + k.
    const bool store_draw = (i >= anneal_end);
    const arma::uword store_row =
      store_draw ? static_cast<arma::uword>(i - anneal_end) : 0;

    const bool constrained_phase = (i >= constrained_start);

    // --- 0. Annealing weights for THIS iteration
    // Computed before anything else, because log_current below has to be
    // evaluated on the same temperature scale the acceptance ratio will use.
    if(constrained_phase && i < anneal_end){
      const double frac = static_cast<double>(i - constrained_start) / anneal_len;
      likelihood_temp = 0.5 + (1.0 - 0.5) * frac;   // 0.5 -> 1
      prior_weight    = 1.7 + (1.0 - 1.7) * frac;   // 1.7 -> 1
    } else {
      likelihood_temp = 1.0;
      prior_weight    = 1.0;
    }

    // --- 1. gamma update
    const double n_off_diag =
      static_cast<double>(p) * (static_cast<double>(p) - 1.0);
    const double n_edges = arma::accu(Adjacency_matrix);   // computed once

    double a = a_gamma + n_edges;
    double b = b_gamma + n_off_diag - n_edges;

    // The raw draw gets its own name; gamma_result stays the outer variable
    // and is assigned (not redeclared) below.
    const double gamma_result_sampled = rbeta_cpp(1, a, b)(0);

    if(!constrained_phase){
      const double frac    = static_cast<double>(i) / burn_in_iterations;
      const double g_floor = 0.02;   // very sparse at start
      gamma_result = g_floor + (gamma_result_sampled - g_floor) * frac;
    } else {
      gamma_result = gamma_result_sampled;   // free after burn-in
    }

    if(store_draw) gamma_list(store_row) = gamma_result;

    // --- 1b. One-time projection at the unconstrained -> constrained handoff
    // WITHOUT THIS the constrained phase deadlocks. The unconstrained phase
    // is free to build overlapping cycles, so the state arriving at
    // constrained_start almost always violates the disjoint-cycle condition.
    if(constrained_phase && !projected){
      projected         = true;
      project_state(Adjacency_matrix, Causal_effect_matrix);
      compute_mixture_mean_var(Z_matrix, mu_mat, tao_mat, p, uN, Mu_full, Tao_full);
      accu_log_Tao = arma::accu(arma::log(Tao_full));
    }

    // gamma_result just changed, so the score has to be re-derived. The
    // incremental rebuild below does that; the old full call here was
    // computing a value nothing read any more.
    if(!inc::inc_rebuild(S, data_matrix, Adjacency_matrix,
                         Causal_effect_matrix, Mu_full, Tao_full)){
      Rcpp::stop("(I - B) is singular or non-invertible at iteration %d; "
                   "cannot score the model.", i);
    }

    // Resynchronise the stability cache and the disjoint-cycle precondition
    // from the current matrices. The projection above is the only place B or
    // Adj change outside a proposal, so doing this once per iteration, here,
    // keeps both exact for the sweeps that follow.
    stab.refresh_all(Causal_effect_matrix);
    topo_valid = !constrained_phase || has_disjoint_cycles(Adjacency_matrix);
    log_current = inc::inc_logpost(S, static_cast<double>(uN),
                                   static_cast<double>(p), accu_log_Tao,
                                   gamma_1, gamma_result,
                                   likelihood_temp, prior_weight);

    // --- 2. Structure + causal effect MH update (add/remove edge)
    double proposal_mean_add = 0;
    double proposal_sd_add;

    if(i < burn_in){
      proposal_sd_add = 0.15;
    } else{
      proposal_sd_add = std::sqrt(gamma_1);
    }

    if(!constrained_phase){
      // No disjoint-cycle constraint in this phase: stability only. Toggles
      // the single candidate cell in place and scores it with an O(N) delta,
      // instead of deep-copying both p x p matrices per candidate and
      // rescoring from scratch.
      double log_q_fwd = 0;
      double log_q_rev = 0;

      for(arma::uword i6 = 0; i6 < p; i6++){

        arma::uvec entries_order = arma::shuffle(remaining_indices[i6]);

        for(arma::uword j6 = 0; j6 < entries_order.n_elem; j6++){
          log_q_fwd = 0;
          log_q_rev = 0;

          arma::uword current_column = entries_order[j6];

          const double old_adj = Adjacency_matrix(i6, current_column);
          const double old_b   = Causal_effect_matrix(i6, current_column);

          double new_adj, new_b;

          if(old_adj == 0){
            new_adj = 1;
            new_b   = proposal_mean_add + proposal_sd_add * arma::randn();
            log_q_fwd = fast_dnorm_log(new_b, proposal_mean_add, proposal_sd_add);
            log_q_rev = 0;
          } else {
            new_adj = 0;
            new_b   = 0;
            log_q_fwd = 0;
            log_q_rev = fast_dnorm_log(old_b, proposal_mean_add, proposal_sd_add);
          }

          Causal_effect_matrix(i6, current_column) = new_b;

          // A proposal is accepted iff it is stable AND passes MH, so the
          // order of the two tests does not change the acceptance
          // probability of any proposal -- unstable proposals are still
          // always rejected.
          inc::Delta D = inc::inc_edge_delta(
            S, data_matrix, Mu_full, Tao_full,
            i6, current_column, old_b, new_b, old_adj, new_adj);

          if(!D.ok){   // (I - B_prop) singular
            Causal_effect_matrix(i6, current_column) = old_b;
            continue;
          }

          const double d_post = inc::inc_delta_logpost(
            D, static_cast<double>(uN), gamma_1, gamma_result,
            likelihood_temp, prior_weight);

          const double log_r = d_post + log_q_rev - log_q_fwd;
          // Compared in log space: exp() of a strongly favoured proposal
          // overflows to inf, and allocating a 2-vector per candidate just
          // to take a min is pure heap traffic in the hottest loop.
          if(std::log(arma::randu()) < log_r
               && stab.check(Causal_effect_matrix, i6, current_column, old_b)){
            Adjacency_matrix(i6, current_column) = new_adj;
            inc::inc_apply(S, D);
            log_current += d_post;
          } else {
            Causal_effect_matrix(i6, current_column) = old_b;
          }
        }
      }
    } else{

      double log_q_fwd = 0;
      double log_q_rev = 0;

      for(arma::uword i6 = 0; i6 < p; i6++){
        // COMPILE ERROR in the original: this was arma::uvec
        // remaining_indices = all_indices.elem(...);
        // arma::shuffle(remaining_indices[i6]); pasted from BCD_old_cpp,
        // where remaining_indices is a std::vector<arma::uvec> so [i6]
        // yields a uvec.
        arma::uvec entries_order = arma::shuffle(remaining_indices[i6]);

        for(arma::uword j6 = 0; j6 < entries_order.n_elem; j6++){
          log_q_fwd = 0;
          log_q_rev = 0;

          arma::uword current_column = entries_order[j6];

          // Toggle the single candidate cell in place instead of copying the
          // full p x p matrices for every candidate edge.
          double old_adj = Adjacency_matrix(i6, current_column);
          double old_b   = Causal_effect_matrix(i6, current_column);

          if(old_adj == 0){
            Adjacency_matrix(i6, current_column) = 1;
            double proposal_coef = proposal_mean_add
            + proposal_sd_add * arma::randn();
            Causal_effect_matrix(i6, current_column) = proposal_coef;
            log_q_fwd = fast_dnorm_log(
              proposal_coef, proposal_mean_add, proposal_sd_add);
            log_q_rev = 0;
          } else {
            Adjacency_matrix(i6, current_column) = 0;
            Causal_effect_matrix(i6, current_column) = 0;
            log_q_fwd = 0;
            log_q_rev = fast_dnorm_log(
              old_b, proposal_mean_add, proposal_sd_add);
          }

          // Disjoint-cycle check first (now O(V+E) with an early exit, see
          // disjoint_cycles_ok above), then stability. Both short-circuit
          // before the O(N) delta.
          if(disjoint_cycles_ok(i6, current_column, old_adj)){
            // O(N) delta instead of a full O(N p^2 + p^3) rescore.
            const double new_b = Causal_effect_matrix(i6, current_column);
            inc::Delta D = inc::inc_edge_delta(
              S, data_matrix, Mu_full, Tao_full,
              i6, current_column, old_b, new_b,
              old_adj, Adjacency_matrix(i6, current_column));

            bool accepted = false;
            if(D.ok){
              const double d_post = inc::inc_delta_logpost(
                D, static_cast<double>(uN), gamma_1, gamma_result,
                likelihood_temp, prior_weight);
              const double log_r = d_post + log_q_rev - log_q_fwd;
              if(std::log(arma::randu()) < log_r
                   && stab.check(Causal_effect_matrix, i6, current_column, old_b)){
                inc::inc_apply(S, D);
                log_current += d_post;
                accepted = true;
              }
            }

            if(!accepted){
              // rejected (or singular): revert just the one cell
              Adjacency_matrix(i6, current_column)     = old_adj;
              Causal_effect_matrix(i6, current_column) = old_b;
            }
          } else {
            // overlapping-cycle proposal: never evaluated, revert the cell
            Adjacency_matrix(i6, current_column)     = old_adj;
            Causal_effect_matrix(i6, current_column) = old_b;
          }
        }
      }
    }

    // 3. Causal effect weight MH update NOTE: this step only reweights
    // EXISTING edges, never adds/removes one — so it can't create or destroy
    // a cycle, and never needs the has_disjoint_cycles() check, only
    // is_stable().
    double burn_in_sd     = 15000;
    arma::vec v           = {static_cast<double>(i), burn_in_sd};
    double weight_prop_sd = 0.03 + 0.07 * (arma::min(v) / burn_in_sd);

    // log_current is already maintained incrementally by step 2. A weight
    // move leaves the adjacency untouched, so the Bernoulli term contributes
    // d_count1 = 0 and cancels, exactly as before.

    for(arma::uword i6 = 0; i6 < p; i6++){
      arma::uvec present_rows = arma::find(Adjacency_matrix.row(i6) != 0);

      if(present_rows.n_elem != 0){
        for(arma::uword j6 = 0; j6 < present_rows.n_elem; j6++){
          arma::uword col = present_rows(j6);
          double old_b = Causal_effect_matrix(i6, col);
          Causal_effect_matrix(i6, col) = old_b + weight_prop_sd * arma::randn();

          {
            // Same single-cell delta machinery: a weight move changes
            // B(i6,col) and nothing else, so Adj is passed unchanged on both
            // sides.
            const double new_b = Causal_effect_matrix(i6, col);
            const double adj_v = Adjacency_matrix(i6, col);

            inc::Delta D = inc::inc_edge_delta(
              S, data_matrix, Mu_full, Tao_full,
              i6, col, old_b, new_b, adj_v, adj_v);

            bool accepted = false;
            if(D.ok){
              const double d_post = inc::inc_delta_logpost(
                D, static_cast<double>(uN), gamma_1, gamma_result,
                likelihood_temp, prior_weight);
              if(std::log(arma::randu()) < d_post
                   && stab.check(Causal_effect_matrix, i6, col, old_b)){
                inc::inc_apply(S, D);
                log_current += d_post;
                accepted = true;
              }
            }
            if(!accepted) Causal_effect_matrix(i6, col) = old_b;
          }
        }
      }
    }

    if(store_draw) Adjacency_matrix_list.row(store_row) =
      arma::vectorise(Adjacency_matrix).t();
    if(store_draw) Causal_effect_matrix_list.row(store_row) =
      arma::vectorise(Causal_effect_matrix).t();

    // 4. Epsilon update
    epsilon_mat = ((arma::eye(p, p) - Causal_effect_matrix)
                     * data_matrix.t()).t();

    // 5. mu update
    for(int i2 = 0; i2 < p; i2++){
      arma::mat first_part = Z_matrix.rows(
        i2 * uN, (i2 + 1) * uN - 1);
      first_part.each_row() /= tao_mat.row(i2);

      for(int j2 = 0; j2 < uM; j2++)
        numerator_result(j2) = (a_mu / b_mu)
        + arma::dot(first_part.col(j2), epsilon_mat.col(i2));

      arma::vec denominator_result = (1.0 / b_mu)
        + arma::sum(first_part, 0).t();
      arma::vec mean_vec = numerator_result / denominator_result;
      arma::vec sd_vec   = arma::sqrt(1.0 / denominator_result);
      arma::rowvec mu_row = mean_vec.t()
        + arma::randn<arma::rowvec>(uM) % sd_vec.t();
      // NOTE: original does not sort mu_row in BCD, preserved as-is
      mu_mat.row(i2) = mu_row;
    }
    if(store_draw) mu_matrix_list.row(store_row) = arma::vectorise(mu_mat).t();

    // 6. tao update
    for(int i4 = 0; i4 < p; i4++){
      arma::mat Z_portion = Z_matrix.rows(
        i4 * uN, (i4 + 1) * uN - 1);
      arma::rowvec a_tao_vec = a_tao + arma::sum(Z_portion, 0) / 2.0;
      for(int j4 = 0; j4 < uM; j4++){
        arma::vec eps_col      = epsilon_mat.col(i4);
        double mu_ij           = mu_mat(i4, j4);
        arma::vec squared_diff = arma::square(eps_col - mu_ij);
        arma::vec b_portion    = 0.5 * Z_portion.col(j4) % squared_diff;
        double b_val           = b_tao + arma::sum(b_portion);
        tao_mat(i4, j4)        = rinvgamma_cpp(1, a_tao_vec(j4), b_val)(0);
      }
    }
    if(store_draw) tao_matrix_list.row(store_row) = arma::vectorise(tao_mat).t();

    // 7. gamma_1 update
    double a_1 = a_gamma_1 + arma::accu(Adjacency_matrix) / 2.0;
    double b_1 = b_gamma_1 + arma::accu(
      Adjacency_matrix % (Causal_effect_matrix % Causal_effect_matrix))
      / 2.0;
    gamma_1 = rinvgamma_cpp(1, a_1, b_1)(0);
    if(store_draw) gamma_1_list(store_row) = gamma_1;

    // 8. Runs every iteration in both phases and is O(N*p*M).
    for(int i5 = 0; i5 < p; i5++){
      for(int j5 = 0; j5 < uM; j5++){
        const double tao_ij = tao_mat(i5, j5);
        z_mu[j5]      = mu_mat(i5, j5);
        z_inv_tao[j5] = 1.0 / tao_ij;
        // log(pi) folded together with the Gaussian normalising constant
        z_const[j5]   = std::log(pi_mat(i5, j5))
          - 0.5 * std::log(tao_ij)
          - 0.5 * std::log(2.0 * M_PI);
      }

      const double* eps_i = epsilon_mat.colptr(i5);

      for(int z5 = 0; z5 < uN; z5++){
        const int iz = i5 * uN + z5;
        const double e = eps_i[z5];

        double max_lp = -std::numeric_limits<double>::infinity();
        for(int j5 = 0; j5 < uM; j5++){
          const double d = e - z_mu[j5];
          z_lp[j5] = z_const[j5] - 0.5 * d * d * z_inv_tao[j5];
          if(z_lp[j5] > max_lp) max_lp = z_lp[j5];
        }

        // Subtracting the max before exp is exactly the stabilisation
        // logSumExp did; the weights are left unnormalised and the total is
        // folded into the uniform draw instead of dividing M times.
        double total = 0.0;
        for(int j5 = 0; j5 < uM; j5++){
          z_w[j5] = std::exp(z_lp[j5] - max_lp);
          total  += z_w[j5];
        }

        const double u = arma::randu() * total;
        double cum_prob = 0.0;
        int sampled     = uM - 1;
        for(int m = 0; m < uM; m++){
          cum_prob += z_w[m];
          if(u < cum_prob){ sampled = m; break; }
        }

        for(int m = 0; m < uM; m++) Z_matrix(iz, m) = 0.0;
        Z_matrix(iz, sampled) = 1.0;
      }
    }

    // 9. pi update
    for(int i6 = 0; i6 < p; i6++){
      arma::mat Z_portion = Z_matrix.rows(
        i6 * uN, (i6 + 1) * uN - 1);
      arma::rowvec Z_colsum     = arma::sum(Z_portion, 0);
      arma::vec alpha_vec_local = Z_colsum.t() + alpha;
      pi_mat.row(i6) = rdirichlet_cpp(alpha_vec_local).t();
    }
    if(store_draw) pi_matrix_list.row(store_row) = arma::vectorise(pi_mat).t();

    // Z_matrix/mu_mat/tao_mat have all changed this iteration (steps 5, 6,
    // 8) — refresh the mixture mean/variance lookup once, here. The next
    // iteration's inc_rebuild reads both, so this is load-bearing, not a
    // diagnostic: do not fold it into anything conditional.
    compute_mixture_mean_var(Z_matrix, mu_mat, tao_mat, p, uN, Mu_full, Tao_full);
    accu_log_Tao = arma::accu(arma::log(Tao_full));

    report_progress(i, n_iter_i, progress_q, "BCD two-phase");

    // Lets Ctrl-C / Esc actually stop the chain. Without this a long run is
    // uninterruptible from the R console.
    if(i % 100 == 0) Rcpp::checkUserInterrupt();
  }

  return List::create(
    Named("Adjacency_matrix_list")     = Adjacency_matrix_list,
    Named("Causal_effect_matrix_list") = Causal_effect_matrix_list,
    Named("gamma_list")                = gamma_list,
    Named("gamma_1_list")              = gamma_1_list,
    Named("mu_matrix_list")            = mu_matrix_list,
    Named("tao_matrix_list")           = tao_matrix_list,
    Named("pi_matrix_list")            = pi_matrix_list,
    // Row k of every trace above is iteration (first_stored_iteration + k).
    Named("first_stored_iteration")    = anneal_end,
    Named("n_stored")                  = n_keep
  );
}
