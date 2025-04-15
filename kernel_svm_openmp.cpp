#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <getopt.h>
#include <optional>
#include <limits>
#include <unordered_set>
#include <omp.h>

const double C = 1.0;
const double poly_c = 1.0;
const int poly_degree = 4;

const double t0 = 1.0;
const double mu = 10.0;
const double tol = 1e-4;
const int newton_iters = 20;
const int barrier_iters = 50;

struct DataPoint {
    Eigen::VectorXd x;
    double y;
};

std::vector<DataPoint> readCSV(const std::string& filename, int max_rows = -1) {
    std::cerr << " Entering readCSV for file: " << filename;
    if (max_rows > 0) {
        std::cerr << ", max_rows=" << max_rows;
    }
    std::cerr << std::endl;
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open file " << filename << std::endl;
        return data;
    }
    int pos = 0;
    int neg = 0;
    std::string line;
    while (std::getline(file, line) && line.empty()) {
        std::cerr << " Skipping empty line" << std::endl;
    }
    
    int row_count = 0;
    while (std::getline(file, line)) {
        if (line.empty()) {
            std::cerr << " Skipping empty line" << std::endl;
            continue;
        }
        
        std::stringstream ss(line);
        std::string value;
        std::vector<double> values;
        
        while (std::getline(ss, value, ',')) {
            try {
                values.push_back(std::stod(value));
            } catch (const std::exception& e) {
                std::cerr << "ERROR: Could not convert value '" << value << "' to double" << std::endl;
                continue;
            }
        }
        
        if (values.size() < 2) {
            std::cerr << "WARNING: Row has too few values, skipping" << std::endl;
            continue;
        }
        
        double label = values.back();
        values.pop_back();
        if (label == 1) {
            pos++;
        } else {
            neg++;
        }
        DataPoint dp;
        dp.x = Eigen::Map<Eigen::VectorXd>(values.data(), values.size());
        dp.y = label;

        data.push_back(dp);
        row_count++;
        
        if (max_rows > 0 && row_count >= max_rows) {
            std::cerr << " Reached maximum number of rows (" << max_rows << ")" << std::endl;
            break;
        }
    }
    
    std::cerr << " Read " << data.size() << " data points from " << filename << std::endl;
    std::cerr << " Positive examples: " << pos << ", Negative examples: " << neg << std::endl;
    return data;
}

double polyKernel(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    return pow(x.dot(y) + poly_c, poly_degree);
}

Eigen::MatrixXd buildKernelMatrix(const std::vector<DataPoint>& data) {
    int n = data.size();
    Eigen::MatrixXd K(n, n);
    
    #pragma omp parallel for shared(data, K)
    for (int i = 0; i < n; i++) {
        K(i, i) = polyKernel(data[i].x, data[i].x);
        for (int j = 0; j < i; j++) {
            K(i, j) = polyKernel(data[i].x, data[j].x);
            K(j, i) = K(i, j);
        }
    }
    return K;
}

Eigen::VectorXd newtonMethod(
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& gradFunc,
    const std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>& hessFunc,
    const std::function<double(const Eigen::VectorXd&)>& objFunc,
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& x0,
    double tol = 1e-6,
    int maxIter = 100)
{
    Eigen::VectorXd x = x0;
    int n = x.size();
    int p = A.rows();

    const double alpha_ls = 0.1;
    const double beta_ls = 0.5;

    for (int iter = 0; iter < maxIter; ++iter) {
        Eigen::VectorXd grad = gradFunc(x);
        Eigen::MatrixXd hess = hessFunc(x);
        
        Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(n + p, n + p);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + p);
        
        KKT.block(0, 0, n, n) = hess;
        if (p > 0) {
             KKT.block(0, n, n, p) = A.transpose();
             KKT.block(n, 0, p, n) = A;
        }
        rhs.segment(0, n) = -grad;

        Eigen::MatrixXd regularized_KKT = KKT;
        regularized_KKT.block(0,0,n,n) += 1e-9 * Eigen::MatrixXd::Identity(n,n);
        Eigen::VectorXd dxy = regularized_KKT.ldlt().solve(rhs);

        Eigen::VectorXd dx = dxy.head(n);

        double lambda_sq = -grad.dot(dx);

        if (lambda_sq / 2.0 < tol) {
            break;
        }

        double t_ls = 1.0;
        double current_obj = objFunc(x);
        while (t_ls > 1e-9) {
             Eigen::VectorXd x_next = x + t_ls * dx;

             double next_obj = objFunc(x_next);
             if (std::isfinite(next_obj) && next_obj < current_obj + alpha_ls * t_ls * grad.dot(dx)) {
                 break;
             }
             t_ls *= beta_ls;
        }

        x += t_ls * dx;

        if (iter == maxIter - 1) {
             std::cerr << "Warning: Newton method reached maximum iterations." << std::endl;
        }
    }
    
    return x;
}

Eigen::VectorXd interiorQP(
    const Eigen::MatrixXd& Q, const Eigen::VectorXd& c,
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& x0,
    double C_param,
    double tol_param = 1e-4,
    double t0_param = 1.0,
    double mu_param = 10.0,
    int barrier_iters_param = 50,
    int newton_iters_param = 20,
    bool phase1 = false) {

    int n_orig = phase1 ? Q.rows() - 1 : Q.rows();
    int n = Q.rows();
    int m = 2 * n_orig;

    Eigen::VectorXd x = x0;

    const double MIN_BARRIER_TERM_QP = 1e-12;

    auto f0 = [&](const Eigen::VectorXd& current_x) -> double {
        return 0.5 * current_x.dot(Q * current_x) + c.dot(current_x);
    };

    auto grad_f0 = [&](const Eigen::VectorXd& current_x) -> Eigen::VectorXd {
        return Q * current_x + c;
    };

    auto phi = [&](const Eigen::VectorXd& current_x) -> double {
        double sum = 0.0;
        if (phase1) {
            double s = current_x(n_orig);
            for (int i = 0; i < n_orig; ++i) {
                double term1 = s + current_x(i);
                double term2 = s - current_x(i) + C_param;
                if (term1 <= 0 || term2 <= 0) return std::numeric_limits<double>::infinity();
                sum -= std::log(std::max(term1, MIN_BARRIER_TERM_QP));
                sum -= std::log(std::max(term2, MIN_BARRIER_TERM_QP));
            }
        } else {
            double thread_sum = 0.0;
            #pragma omp parallel for shared(current_x) reduction(+:thread_sum)
            for (int i = 0; i < n_orig; ++i) {
                double term1 = current_x(i);
                double term2 = C_param - current_x(i);
                if (term1 <= 0 || term2 <= 0) {
                    thread_sum = std::numeric_limits<double>::infinity();
                } else {
                    thread_sum -= std::log(std::max(term1, MIN_BARRIER_TERM_QP));
                    thread_sum -= std::log(std::max(term2, MIN_BARRIER_TERM_QP));
                }
            }
            sum = thread_sum;
        }
        return sum;
    };

    auto grad_phi = [&](const Eigen::VectorXd& current_x) -> Eigen::VectorXd {
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(n);
        if (phase1) {
            double s = current_x(n_orig);
            double grad_s = 0.0;
            for (int i = 0; i < n_orig; ++i) {
                double term1 = std::max(s + current_x(i), MIN_BARRIER_TERM_QP);
                double term2 = std::max(s - current_x(i) + C_param, MIN_BARRIER_TERM_QP);
                grad(i) = -1.0 / term1 + 1.0 / term2;
                grad_s += -1.0 / term1 - 1.0 / term2;
            }
            grad(n_orig) = grad_s;
        } else {
            #pragma omp parallel for shared(current_x)
            for (int i = 0; i < n_orig; ++i) {
                double term1 = std::max(current_x(i), MIN_BARRIER_TERM_QP);
                double term2 = std::max(C_param - current_x(i), MIN_BARRIER_TERM_QP);
                grad(i) = -1.0 / term1 + 1.0 / term2;
            }
        }
        return grad;
    };

    auto hess_phi = [&](const Eigen::VectorXd& current_x) -> Eigen::MatrixXd {
        Eigen::MatrixXd H_phi = Eigen::MatrixXd::Zero(n, n);
        if (phase1) {
            double s = current_x(n_orig);
            double H_ss = 0.0;
            for (int i = 0; i < n_orig; ++i) {
                double term1 = std::max(s + current_x(i), MIN_BARRIER_TERM_QP);
                double term2 = std::max(s - current_x(i) + C_param, MIN_BARRIER_TERM_QP);
                double H_ii = 1.0 / (term1 * term1) + 1.0 / (term2 * term2);
                double H_is = 1.0 / (term1 * term1) - 1.0 / (term2 * term2);
                H_phi(i, i) = H_ii;
                H_phi(i, n_orig) = H_is;
                H_phi(n_orig, i) = H_is;
                H_ss += H_ii;
            }
             H_phi(n_orig, n_orig) = H_ss;
        } else {
            #pragma omp parallel for shared(current_x)
            for (int i = 0; i < n_orig; ++i) {
                double term1 = std::max(current_x(i), MIN_BARRIER_TERM_QP);
                double term2 = std::max(C_param - current_x(i), MIN_BARRIER_TERM_QP);
                H_phi(i, i) = 1.0 / (term1 * term1) + 1.0 / (term2 * term2);
            }
        }
        return H_phi;
    };

    double t = t0_param;

    for (int iter = 0; iter < barrier_iters_param; iter++) {
        std::cerr << " Barrier iter " << iter << std::endl;
        auto combined_objective = [&](const Eigen::VectorXd& current_x) -> double {
            double f0_val = f0(current_x);
            double phi_val = phi(current_x);
            if (!std::isfinite(f0_val) || !std::isfinite(phi_val)) {
                 return std::numeric_limits<double>::infinity();
            }
            return t * f0_val + phi_val;
        };

        auto combined_gradient = [&](const Eigen::VectorXd& current_x) -> Eigen::VectorXd {
            return t * grad_f0(current_x) + grad_phi(current_x);
        };

        auto combined_hessian = [&](const Eigen::VectorXd& current_x) -> Eigen::MatrixXd {
            return t * Q + hess_phi(current_x);
        };

        double newton_tol = 1e-6;
        x = newtonMethod(
            combined_gradient,
            combined_hessian,
            combined_objective,
            A, x,
            newton_tol,
            newton_iters_param
        );

        double gap = static_cast<double>(m) / t;
        if (gap < tol_param) {
            break;
        }

        t *= mu_param;

         if (iter == barrier_iters_param - 1) {
             std::cerr << "Warning: Barrier method reached maximum iterations." << std::endl;
         }
    }

    return x;
}

std::optional<Eigen::VectorXd> phase_1(
    const Eigen::MatrixXd& A_orig,
    const Eigen::VectorXd& b_orig,
    double C_param,
    int n_orig,
    double tol_param = 1e-6,
    double t0_param = 1.0,
    double mu_param = 10.0,
    int barrier_iters_param = 50,
    int newton_iters_param = 20,
    double phase1_gap_tol = 1e-4)
{
    int p = A_orig.rows();

    Eigen::VectorXd x_init;
    if (p == 0) {
        x_init = Eigen::VectorXd::Constant(n_orig, C_param / 2.0);
    } else {
        Eigen::VectorXd x_guess = Eigen::VectorXd::Constant(n_orig, C_param / 2.0);
        Eigen::MatrixXd AAT = A_orig * A_orig.transpose();
        AAT += 1e-9 * Eigen::MatrixXd::Identity(p, p);
        Eigen::VectorXd correction = A_orig.transpose() * AAT.ldlt().solve(A_orig * x_guess - b_orig);
        x_init = x_guess - correction;
    }

    Eigen::MatrixXd Q_p1 = Eigen::MatrixXd::Zero(n_orig + 1, n_orig + 1);
    Eigen::VectorXd c_p1 = Eigen::VectorXd::Zero(n_orig + 1);
    c_p1(n_orig) = 1.0;

    Eigen::MatrixXd A_p1(p, n_orig + 1);
    A_p1.leftCols(n_orig) = A_orig;
    A_p1.rightCols(1) = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd b_p1 = b_orig;

    double s_init = 1.0;
    for(int i=0; i<n_orig; ++i) {
        s_init = std::max(s_init, -x_init(i) + 1e-6);
        s_init = std::max(s_init, x_init(i) - C_param + 1e-6);
    }

    Eigen::VectorXd z_init(n_orig + 1);
    z_init.head(n_orig) = x_init;
    z_init(n_orig) = s_init;

    Eigen::VectorXd z_opt = interiorQP(
        Q_p1, c_p1, A_p1, z_init,
        C_param,
        phase1_gap_tol,
        t0_param, mu_param,
        barrier_iters_param, newton_iters_param,
        true
    );

    double optimal_s = z_opt(n_orig);

    if (optimal_s < -tol_param) {
        Eigen::VectorXd x_feasible = z_opt.head(n_orig);

        if (p > 0 && (A_orig * x_feasible - b_orig).norm() > 1e-6 * std::max(1.0, b_orig.norm())) {
             std::cerr << "Warning: Phase 1 solution does not satisfy Ax=b accurately. Norm: "
                       << (A_orig * x_feasible - b_orig).norm() << std::endl;
        }

        bool strictly_feasible_bounds = true;
        for(int i=0; i<n_orig; ++i) {
            if (x_feasible(i) <= tol_param || x_feasible(i) >= C_param - tol_param) {
                 std::cerr << "Warning: Phase 1 solution x_feasible(" << i << ") = " << x_feasible(i)
                           << " is not strictly between 0+" << tol_param << " and C-" << tol_param << " (C=" << C_param << ")" << std::endl;
                 strictly_feasible_bounds = false;
            }
        }

        if (!strictly_feasible_bounds) {
             std::cerr << "Error: Phase 1 solution not strictly feasible for box constraints. Cannot proceed to Phase 2." << std::endl;
             return std::nullopt;
        }

        return x_feasible;

    } else if (optimal_s < tol_param) {
         std::cerr << "Warning: Phase 1 indicates feasibility, but possibly not strict feasibility (s ~= 0: " << optimal_s << "). Problem might be difficult." << std::endl;
         return std::nullopt;
    }
     else {
        std::cerr << "Error: Phase 1 failed. Problem may be infeasible (s > 0: " << optimal_s << ")." << std::endl;
        return std::nullopt;
    }
}

double computeBias(const std::vector<DataPoint>& data, const Eigen::VectorXd& alpha, const Eigen::MatrixXd& K) {
    double bias = 0;
    int count = 0; 
    #pragma omp parallel for shared(data, alpha, K) reduction(+:bias) reduction(+:count)
    for (size_t i = 0; i < data.size(); i++) {
        if(alpha(i) > 1e-5 && alpha(i) < C - 1e-5) {
            for (size_t j = 0; j < data.size(); j++) {
                bias -= alpha(j) * data[j].y * K(i, j);
            }
            bias += data[i].y;
            count++;
        }
    }
    if(count == 0) {
        return 0.0;
    }
    return bias / count;
}

double calculateAccuracy(const std::vector<DataPoint>& train_data,
                         const std::vector<DataPoint>& test_data,
                         const Eigen::VectorXd& alpha,
                         double bias) {
    if (test_data.empty()) {
        std::cerr << "Error: No test data loaded" << std::endl;
        return 0.0;
    }
    if (train_data.empty() || static_cast<size_t>(alpha.size()) != train_data.size()) {
         std::cerr << "Error: Training data size mismatch for accuracy calculation." << std::endl;
         return 0.0;
    }

    size_t N_train = train_data.size();
    size_t N_test = test_data.size();
    int correct = 0;

    #pragma omp parallel for shared(train_data, test_data, alpha, bias) reduction(+:correct)
    for (size_t i = 0; i < N_test; ++i) {
        double decision_value = 0.0;
        
        for (size_t j = 0; j < N_train; ++j) {
            decision_value += alpha(j) * train_data[j].y * polyKernel(test_data[i].x, train_data[j].x);
        }
        decision_value += bias;

        if ((decision_value >= 0 && test_data[i].y > 0) ||
            (decision_value < 0 && test_data[i].y < 0)) {
            correct++;
        }
    }

    double accuracy = static_cast<double>(correct) / N_test;
    return accuracy;
}

int main(int argc, char* argv[]) {
    std::string train_file = "train_data.csv";
    std::string test_file = "test_data.csv";
    int num_rows = 10;
    double svm_C = C;
    double main_tol = tol;
    double phase1_tol = 1e-6;
    double phase1_gap_tol = 1e-4;
    int test_size = 10;
    int num_threads = omp_get_max_threads();

    int opt;
    while ((opt = getopt(argc, argv, "r:s:t:")) != -1) {
        switch (opt) {
            case 'r':
                num_rows = std::stoi(optarg);
                break;
            case 's':
                test_size = std::stoi(optarg);
                break;
            case 't':
                num_threads = std::stoi(optarg);
                omp_set_num_threads(num_threads);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-r num_rows] [-s test_size] [-t num_threads]" << std::endl;
                return 1;
        }
    }

    std::cout << "Running with " << num_threads << " threads" << std::endl;

    try {
        std::vector<DataPoint> data = readCSV(train_file, num_rows);
        if(data.empty()) {
            std::cerr << "Error: No training data loaded" << std::endl;
            return 1;
        }

        auto total_start = std::chrono::high_resolution_clock::now();

        int N = data.size();      
        Eigen::MatrixXd K = buildKernelMatrix(data);
        Eigen::VectorXd y_vec(N);

        for (int i = 0; i < N; i++) {
            y_vec(i) = data[i].y;
        }

        Eigen::MatrixXd Y = y_vec * y_vec.transpose();
        Eigen::MatrixXd Q_svm = K.array() * Y.array();
        Q_svm += 1e-9 * Eigen::MatrixXd::Identity(N, N);

        Eigen::VectorXd c_svm = -Eigen::VectorXd::Ones(N);

        Eigen::MatrixXd A_svm(1, N);
        A_svm.row(0) = y_vec.transpose();

        Eigen::VectorXd b_svm(1);
        b_svm(0) = 0.0;

        auto train_start = std::chrono::high_resolution_clock::now();

        std::optional<Eigen::VectorXd> alpha0_opt = phase_1(
            A_svm, b_svm, svm_C, N,
            phase1_tol, 
            t0, mu, barrier_iters, newton_iters,
            phase1_gap_tol
        );

        if (!alpha0_opt) {
            std::cerr << "Error: Phase I failed to find a strictly feasible starting point." << std::endl;
            bool all_pos = (y_vec.array() > 0).all();
            bool all_neg = (y_vec.array() < 0).all();
            if (all_pos || all_neg) {
                 std::cerr << "Hint: All labels have the same sign." << std::endl;
            }
            return 1;
        }
        Eigen::VectorXd alpha0 = *alpha0_opt;

        Eigen::VectorXd alpha = interiorQP(
            Q_svm, c_svm, A_svm, alpha0,
            svm_C,
            main_tol,
            t0, mu,
            barrier_iters, newton_iters,
            false
        );

        auto train_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> train_time = train_end - train_start;

        double bias = computeBias(data, alpha, K);

        std::cout << "Loading test data and computing accuracy..." << std::endl;
        auto test_start = std::chrono::high_resolution_clock::now();
        std::vector<DataPoint> test_data = readCSV(test_file, test_size);
        double accuracy = calculateAccuracy(data, test_data, alpha, bias);
        auto test_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> test_time = test_end - test_start;
        std::cout << "Testing completed." << std::endl;

        std::string output_filename = "result_openmp/" + std::to_string(num_rows) + "_" + std::to_string(test_size) + "_" + std::to_string(num_threads);
        std::ofstream output_file(output_filename);
        std::ofstream output_file_global("experiment_results.txt", std::ios::app);

        if (output_file.is_open()) {
            for (int i = 0; i < alpha.size(); i++) {
                output_file << alpha[i] << " ";
            }
            output_file << "\n";
            output_file << bias << "\n";
            output_file << train_time.count() << "\n";
            output_file << test_time.count() << "\n";
            output_file << accuracy << "\n";
            output_file.close();
            output_file_global << "OpenMP: "
            << "Train Rows:" << num_rows
                        << ", Test Rows:" << test_size
                        << ", Train Time:" << train_time.count()
                        << ", Test Time:" << test_time.count()
                        << ", Accuracy:" << accuracy
                        <<  "Threads:" << num_threads <<"\n";
            output_file_global.close();
            std::cout << "OpenMP results for " << num_rows << " rows, test size " << test_size << ", threads " << num_threads << ":" << std::endl;
            std::cout << "Total Training time: " << train_time.count() << " seconds" << std::endl;
            std::cout << "Test time: " << test_time.count() << " seconds" << std::endl;
            std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
            std::cout << "Bias: " << bias << std::endl;
        } else {
            std::cerr << "Error: Unable to open output file: " << output_filename << std::endl;
            return 1;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: Exception caught in main: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Error: Unknown exception caught in main." << std::endl;
        return 1;
    }

    return 0;
}