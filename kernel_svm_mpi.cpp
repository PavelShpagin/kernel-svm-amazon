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
#include <mpi.h>
#include <cstring>
#include <numeric>
#include <iomanip>

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

std::pair<int, int> calculate_chunk(int total_size, int num_procs, int rank) {
    if (total_size == 0) {
        return {0, 0};
    }
    int base_chunk_size = total_size / num_procs;
    int remainder = total_size % num_procs;
    int my_start_index = rank * base_chunk_size;
    int my_count = base_chunk_size + (rank == num_procs - 1 ? remainder : 0);
    return {my_start_index, my_count};
}

Eigen::MatrixXd buildKernelMatrix(const std::vector<DataPoint>& data, int rank) {
    int n = data.size();
    Eigen::MatrixXd K(n, n);
    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            K(i, i) = polyKernel(data[i].x, data[i].x);
            for (int j = 0; j < i; j++) {
                K(i, j) = polyKernel(data[i].x, data[j].x);
                K(j, i) = K(i, j);
            }
        }
    }
    return K;
}

// Eigen::MatrixXd buildKernelMatrix_MPI(const std::vector<DataPoint>& data, MPI_Comm comm, int rank, int size) {
//     int n = data.size();
//     Eigen::MatrixXd K(n, n);
//     for (int i = 0; i < n; i++) {
//         K(i, i) = polyKernel(data[i].x, data[i].x);
//         for (int j = 0; j < i; j++) {
//             K(i, j) = polyKernel(data[i].x, data[j].x);
//             K(j, i) = K(i, j);
//         }
//     }
//     return K;
// }

Eigen::VectorXd newtonMethod(
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&, MPI_Comm, int, int)>& gradFunc,
    const std::function<Eigen::MatrixXd(const Eigen::VectorXd&, MPI_Comm, int, int)>& hessFunc,
    const std::function<double(const Eigen::VectorXd&, MPI_Comm, int, int)>& objFunc,
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& x0,
    double tol = 1e-6,
    int maxIter = 100,
    MPI_Comm comm = MPI_COMM_WORLD,
    int rank = 0,
    int size = 1)
{
    Eigen::VectorXd x = x0;
    int n = x.size();
    int p = A.rows();

    const double alpha_ls = 0.1;
    const double beta_ls = 0.5;

    Eigen::MatrixXd KKT;
    Eigen::VectorXd rhs;
    Eigen::VectorXd dx;
    Eigen::VectorXd dxy;
    Eigen::VectorXd x_next = Eigen::VectorXd::Zero(n);
    double lambda_sq = 0;
    double t_ls = 0;
    int break_flag = 0;

    for (int iter = 0; iter < maxIter; ++iter) {
        MPI_Bcast(x.data(), x.size(), MPI_DOUBLE, 0, comm);
        Eigen::VectorXd grad = gradFunc(x, comm, rank, size);
        Eigen::MatrixXd hess = hessFunc(x, comm, rank, size);
        if (rank == 0) {
        
            KKT = Eigen::MatrixXd::Zero(n + p, n + p);
            rhs = Eigen::VectorXd::Zero(n + p);
            
            KKT.block(0, 0, n, n) = hess;
            if (p > 0) {
                KKT.block(0, n, n, p) = A.transpose();
                KKT.block(n, 0, p, n) = A;
            }
            rhs.segment(0, n) = -grad;

            Eigen::MatrixXd regularized_KKT = KKT;
            regularized_KKT.block(0,0,n,n) += 1e-9 * Eigen::MatrixXd::Identity(n,n);
            dxy = regularized_KKT.ldlt().solve(rhs);

            dx = dxy.head(n);

            lambda_sq = -grad.dot(dx);

            if (lambda_sq / 2.0 < tol) {
                break_flag = 1;
            }
        }

        MPI_Bcast(&break_flag, 1, MPI_INT, 0, comm);
        if (break_flag) {
            break;
        }

        t_ls = 1.0;

        double current_obj = objFunc(x, comm, rank, size);

        while (t_ls > 1e-9) {
            if (rank == 0) {
                x_next = x + t_ls * dx;
            }

            MPI_Bcast(x_next.data(), x_next.size(), MPI_DOUBLE, 0, comm);

            double next_obj = objFunc(x_next, comm, rank, size);

            if (rank == 0) {
                if (std::isfinite(next_obj) && next_obj < current_obj + alpha_ls * t_ls * grad.dot(dx)) {
                    // x = x_next;
                    break_flag = 1;
                }
            }

            MPI_Bcast(&break_flag, 1, MPI_INT, 0, comm);
            
            if (break_flag) {
                break;
            }
            t_ls *= beta_ls;
        }

        if (rank == 0) {
            x += t_ls * dx;       
        }
        
        if (rank == 0 && iter == maxIter - 1) {
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
    bool phase1 = false,
    MPI_Comm comm = MPI_COMM_WORLD,
    int rank = 0,
    int size = 1) {

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

    auto phi = [&](const Eigen::VectorXd& current_x, MPI_Comm comm, int rank, int size) -> double {
        double sum = 0.0;
        double sum_global = 0.0;

        std::pair<int, int> chunk = calculate_chunk(n_orig, size, rank);
        int start_idx = chunk.first;
        int count_local = chunk.second;

        if (phase1) {
            double s = current_x(n_orig);
            for (int i = start_idx; i < start_idx + count_local; ++i) {
                double term1 = s + current_x(i);
                double term2 = s - current_x(i) + C_param;
                if (term1 <= 0 || term2 <= 0) {
                    sum = std::numeric_limits<double>::infinity();
                } else {
                    sum -= std::log(std::max(term1, MIN_BARRIER_TERM_QP));
                    sum -= std::log(std::max(term2, MIN_BARRIER_TERM_QP));
                }
            }
        } else {
            for (int i = start_idx; i < start_idx + count_local; ++i) {
                double term1 = current_x(i);
                double term2 = C_param - current_x(i);
                if (term1 <= 0 || term2 <= 0) {
                    sum = std::numeric_limits<double>::infinity();
                } else {
                    sum -= std::log(std::max(term1, MIN_BARRIER_TERM_QP));
                    sum -= std::log(std::max(term2, MIN_BARRIER_TERM_QP));
                }
            }
        }
        MPI_Reduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
        return sum_global;
    };

    auto grad_phi = [&](const Eigen::VectorXd& current_x, MPI_Comm comm, int rank, int size) -> Eigen::VectorXd {
        Eigen::VectorXd grad_result = Eigen::VectorXd::Zero(n);

        std::pair<int, int> chunk = calculate_chunk(n_orig, size, rank);
        int count_local = chunk.second;
        int start_idx_global = chunk.first;

        Eigen::VectorXd grad_local = Eigen::VectorXd::Zero(count_local);
        double grad_s_local = 0.0;

        if (count_local > 0) {
            if (phase1) {
                double s = current_x(n_orig);
                for (int i = 0; i < count_local; ++i) {
                    int global_idx = start_idx_global + i;
                    double term1 = std::max(s + current_x(global_idx), MIN_BARRIER_TERM_QP);
                    double term2 = std::max(s - current_x(global_idx) + C_param, MIN_BARRIER_TERM_QP);
                    grad_local(i) = -1.0 / term1 + 1.0 / term2;
                    grad_s_local += -1.0 / term1 - 1.0 / term2;
                }
            } else {
                for (int i = 0; i < count_local; ++i) {
                    int global_idx = start_idx_global + i;
                    double term1 = std::max(current_x(global_idx), MIN_BARRIER_TERM_QP);
                    double term2 = std::max(C_param - current_x(global_idx), MIN_BARRIER_TERM_QP);
                    grad_local(i) = -1.0 / term1 + 1.0 / term2;
                }
            }
        }

        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        if (rank == 0) {
            int current_displ = 0;
            for (int i = 0; i < size; ++i) {
                std::pair<int, int> proc_chunk = calculate_chunk(n_orig, size, i);
                recvcounts[i] = proc_chunk.second;
                displs[i] = current_displ;
                current_displ += recvcounts[i];
            }
        }

        MPI_Gatherv(grad_local.data(), count_local, MPI_DOUBLE,
                    rank == 0 ? grad_result.data() : nullptr,
                    recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, comm);

        if (phase1) {
            double grad_s_global = 0.0;
            MPI_Reduce(&grad_s_local, &grad_s_global, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
            if (rank == 0) {
                grad_result(n_orig) = grad_s_global;
            }
        }

        return grad_result;
    };

    auto hess_phi = [&](const Eigen::VectorXd& current_x, MPI_Comm comm, int rank, int size) -> Eigen::MatrixXd {
        Eigen::MatrixXd H_phi;
        if (rank == 0) {
            H_phi = Eigen::MatrixXd::Zero(n, n);
        } else {
            H_phi = Eigen::MatrixXd::Zero(0, 0);
        }

        double H_ss_local = 0.0;

        std::pair<int, int> chunk = calculate_chunk(n_orig, size, rank);
        int start_idx_global = chunk.first;
        int count_local = chunk.second;

        Eigen::VectorXd H_ii_local = Eigen::VectorXd::Zero(count_local);
        Eigen::VectorXd H_is_local;
        if (phase1) {
            H_is_local = Eigen::VectorXd::Zero(count_local);
        }

        if (count_local > 0) {
            if (phase1) {
                double s = current_x(n_orig);
                for (int i = 0; i < count_local; ++i) {
                    int global_idx = start_idx_global + i;
                    double term1 = std::max(s + current_x(global_idx), MIN_BARRIER_TERM_QP);
                    double term2 = std::max(s - current_x(global_idx) + C_param, MIN_BARRIER_TERM_QP);
                    double H_ii = 1.0 / (term1 * term1) + 1.0 / (term2 * term2);
                    double H_is = 1.0 / (term1 * term1) - 1.0 / (term2 * term2);

                    H_ii_local(i) = H_ii;
                    H_is_local(i) = H_is;
                    H_ss_local += H_ii;
                }
            } else {
                for (int i = 0; i < count_local; ++i) {
                    int global_idx = start_idx_global + i;
                    double term1 = std::max(current_x(global_idx), MIN_BARRIER_TERM_QP);
                    double term2 = std::max(C_param - current_x(global_idx), MIN_BARRIER_TERM_QP);
                    H_ii_local(i) = 1.0 / (term1 * term1) + 1.0 / (term2 * term2);
                }
            }
        }

        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        if (rank == 0) {
            int current_displ = 0;
            for (int i = 0; i < size; ++i) {
                std::pair<int, int> proc_chunk = calculate_chunk(n_orig, size, i);
                recvcounts[i] = proc_chunk.second;
                displs[i] = current_displ;
                current_displ += recvcounts[i];
            }
        }

        Eigen::VectorXd H_ii_full;
        if (rank == 0) {
            H_ii_full.resize(n_orig);
        }
        MPI_Gatherv(H_ii_local.data(), count_local, MPI_DOUBLE,
                    rank == 0 ? H_ii_full.data() : nullptr,
                    recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, comm);

        if (phase1) {
            double H_ss_global = 0.0;
            MPI_Reduce(&H_ss_local, &H_ss_global, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

            Eigen::VectorXd H_is_full;
             if (rank == 0) {
                H_is_full.resize(n_orig);
            }
            MPI_Gatherv(H_is_local.data(), count_local, MPI_DOUBLE,
                        rank == 0 ? H_is_full.data() : nullptr,
                        recvcounts.data(), displs.data(), MPI_DOUBLE,
                        0, comm);

            if (rank == 0) {
                H_phi.diagonal().head(n_orig) = H_ii_full;
                H_phi.col(n_orig).head(n_orig) = H_is_full;
                H_phi.row(n_orig).head(n_orig) = H_is_full.transpose();
                H_phi(n_orig, n_orig) = H_ss_global;
            }
        } else {
            if (rank == 0) {
                 H_phi.diagonal().head(n_orig) = H_ii_full;
            }
        }
        
        return H_phi;
    };

    double t = t0_param;

    for (int iter = 0; iter < barrier_iters_param; iter++) {
        if (rank == 0) {
            std::cerr << " Barrier iter " << iter << std::endl;
        }

        auto combined_objective = [&](const Eigen::VectorXd& current_x, MPI_Comm comm, int rank, int size) -> double {
            double f0_val = 0.0;
            if (rank == 0) {
                f0_val = f0(current_x);
            }
            double phi_val = phi(current_x, comm, rank, size);
            if (rank == 0) {
                if (!std::isfinite(f0_val) || !std::isfinite(phi_val)) {
                     return std::numeric_limits<double>::infinity();
                }
                return t * f0_val + phi_val;
            }
            return 0.0;
        };

        auto combined_gradient = [&](const Eigen::VectorXd& current_x, MPI_Comm comm, int rank, int size) -> Eigen::VectorXd {
            Eigen::VectorXd grad = Eigen::VectorXd::Zero(n);
            Eigen::VectorXd grad_phi_val = grad_phi(current_x, comm, rank, size);
            if (rank == 0) {
                grad = t * grad_f0(current_x) + grad_phi_val;
            }
            return grad;
        };

        auto combined_hessian = [&](const Eigen::VectorXd& current_x, MPI_Comm comm, int rank, int size) -> Eigen::MatrixXd {
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
            Eigen::MatrixXd H_phi = hess_phi(current_x, comm, rank, size);
            if (rank == 0) {
                H = t * Q + H_phi;
            }
            return H;
        };

        double newton_tol = 1e-6;
        MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, comm);
                
        x = newtonMethod(
            combined_gradient,
            combined_hessian,
            combined_objective,
            A, x,
            newton_tol,
            newton_iters_param,
            comm, rank, size
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
    double phase1_gap_tol = 1e-4,
    MPI_Comm comm = MPI_COMM_WORLD,
    int rank = 0,
    int size = 1)
{
    Eigen::VectorXd x_init;
    Eigen::MatrixXd Q_p1;
    Eigen::VectorXd c_p1;
    Eigen::MatrixXd A_p1;
    Eigen::VectorXd b_p1;
    Eigen::VectorXd z_init;
    double s_init_local = 1.0;
    int p = A_orig.rows();

    if (rank == 0) {
        if (p == 0) {
            x_init = Eigen::VectorXd::Constant(n_orig, C_param / 2.0);
        } else {
            Eigen::VectorXd x_guess = Eigen::VectorXd::Constant(n_orig, C_param / 2.0);
            Eigen::MatrixXd AAT = A_orig * A_orig.transpose();
            AAT += 1e-9 * Eigen::MatrixXd::Identity(p, p);
            Eigen::VectorXd correction = A_orig.transpose() * AAT.ldlt().solve(A_orig * x_guess - b_orig);
            x_init = x_guess - correction;
        }

        Q_p1 = Eigen::MatrixXd::Zero(n_orig + 1, n_orig + 1);
        c_p1 = Eigen::VectorXd::Zero(n_orig + 1);
        c_p1(n_orig) = 1.0;

        A_p1.resize(p, n_orig + 1);
        if (p > 0) {
            A_p1.leftCols(n_orig) = A_orig;
            A_p1.rightCols(1) = Eigen::VectorXd::Zero(p);
        } else {
             A_p1.setZero();
        }
        b_p1 = b_orig;
    } else {
        x_init.resize(n_orig);
        Q_p1.resize(n_orig + 1, n_orig + 1);
        c_p1.resize(n_orig + 1);
        A_p1.resize(p, n_orig + 1);
        b_p1.resize(1);
    }

    MPI_Bcast(x_init.data(), n_orig, MPI_DOUBLE, 0, comm);

    std::pair<int, int> chunk = calculate_chunk(n_orig, size, rank);
    int start_idx = chunk.first;
    int count_local = chunk.second;

    for(int i=start_idx; i<start_idx+count_local; ++i) {
        s_init_local = std::max(s_init_local, -x_init(i) + 1e-6);
        s_init_local = std::max(s_init_local, x_init(i) - C_param + 1e-6);
    }

    double s_init_global = 0.0;
    MPI_Reduce(&s_init_local, &s_init_global, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    z_init.resize(n_orig + 1);
    if (rank == 0) {
        z_init.head(n_orig) = x_init;
        z_init(n_orig) = s_init_global;
    }

    Eigen::VectorXd z_opt = interiorQP(
        Q_p1, c_p1, A_p1, z_init,
        C_param,
        phase1_gap_tol,
        t0_param, mu_param,
        barrier_iters_param, newton_iters_param,
        true,
        comm, rank, size
    );

    if (rank == 0) {
        double optimal_s = z_opt(n_orig);

        if (optimal_s < -tol_param) {
            Eigen::VectorXd x_feasible = z_opt.head(n_orig);

            if (A_orig.rows() > 0 && (A_orig * x_feasible - b_orig).norm() > 1e-5 * std::max(1.0, b_orig.norm())) {
                 std::cerr << "Rank 0 Warning: Phase 1 solution does not satisfy Ax=b accurately. Norm: "
                           << (A_orig * x_feasible - b_orig).norm() << std::endl;
            }

            bool strictly_feasible_bounds = true;
            for(int i=0; i<n_orig; ++i) {
                if (x_feasible(i) <= tol_param || x_feasible(i) >= C_param - tol_param) {
                     std::cerr << "Rank 0 Warning: Phase 1 solution x_feasible(" << i << ") = " << x_feasible(i)
                               << " is not strictly between 0+" << tol_param << " and C-" << tol_param << " (C=" << C_param << ")" << std::endl;
                     strictly_feasible_bounds = false;
                     break;
                }
            }

            if (!strictly_feasible_bounds) {
                 std::cerr << "Rank 0 Error: Phase 1 solution not strictly feasible for box constraints. Cannot proceed to Phase 2." << std::endl;
                 return std::nullopt;
            }

            return x_feasible;

        } else if (optimal_s < tol_param) {
             std::cerr << "Rank 0 Warning: Phase 1 indicates feasibility, but possibly not strict feasibility (s ~= 0: " << optimal_s << "). Problem might be difficult." << std::endl;
             return std::nullopt;
        } else {
            std::cerr << "Rank 0 Error: Phase 1 failed. Problem may be infeasible (s > 0: " << optimal_s << ")." << std::endl;
            return std::nullopt;
        }
    }
    return std::nullopt;
}

double computeBias(const std::vector<DataPoint>& data, const Eigen::VectorXd& alpha, const Eigen::MatrixXd& K) {
    // This function should only be called on rank 0 as K is only computed there.
    std::vector<double> b_vals; // Store individual bias values like in sequential code
    for (size_t i = 0; i < data.size(); i++) {
        // Check if alpha[i] corresponds to a support vector on the margin
        if(alpha(i) > 1e-5 && alpha(i) < C - 1e-5) {
            double sum_kernel_term = 0.0;
            for (size_t j = 0; j < data.size(); j++) {
                // Ensure K(i,j) access is valid (only rank 0 has the full K)
                sum_kernel_term += alpha(j) * data[j].y * K(i, j);
            }
            double b_i = data[i].y - sum_kernel_term; // Calculate the bias for this support vector
            b_vals.push_back(b_i); // Store it
        }
    }
    if(b_vals.empty()) {
        // Add a warning if no suitable support vectors are found
        std::cerr << "Rank 0 Warning: No support vectors found for bias calculation (all alpha near 0 or C)." << std::endl;
        // Handle case where no support vectors are found (e.g., return 0 or calculate differently)
        // For now, returning average over all points might be less meaningful than 0 or an error.
        // Let's compute an average prediction over all points as a fallback, though not standard SVM practice.
        // Or simply return 0. Returning 0 is simpler.
        return 0.0;
    }
    // Calculate the average bias from the collected values
    double sum_b = 0;
    for(double b : b_vals) {
        sum_b += b;
    }
    return sum_b / b_vals.size();
}

double calculateAccuracy_MPI(const std::vector<DataPoint>& train_data,
                             const std::vector<DataPoint>& test_data,
                             const Eigen::VectorXd& alpha,
                             double bias,
                             int N_train, int N_test, int d,
                             int rank, int size, MPI_Comm comm) {
    int local_correct = 0;

    Eigen::VectorXd test_y;
    std::vector<double> test_X_flat;
    test_y.resize(N_test);
    test_X_flat.resize(N_test * d);

    if (rank == 0) {
        for(int i=0; i<N_test; ++i) {
            test_y(i) = test_data[i].y;
            if (test_data[i].x.size() == d) {
                 memcpy(test_X_flat.data() + i * d, test_data[i].x.data(), d * sizeof(double));
            } else {
            if (i==0) std::cerr << "Rank 0 WARNING in calculateAccuracy_MPI: test_data[" << i << "].x.size()=" << test_data[i].x.size() << " != d=" << d << std::endl;
            }
        }
    }
    MPI_Bcast(test_y.data(), N_test, MPI_DOUBLE, 0, comm);
    MPI_Bcast(test_X_flat.data(), N_test * d, MPI_DOUBLE, 0, comm);

    std::pair<int, int> chunk = calculate_chunk(N_test, size, rank);
    int start_idx = chunk.first;
    int count_local = chunk.second;

    Eigen::VectorXd current_test_x(d);

    for (int i = start_idx; i < start_idx + count_local; ++i) {
        double decision_value = 0.0;
        current_test_x = Eigen::Map<const Eigen::VectorXd>(test_X_flat.data() + i * d, d);

        for (size_t j = 0; j < static_cast<size_t>(N_train); ++j) {
            if (std::abs(alpha(j)) > 1e-9) {
                decision_value += alpha(j) * train_data[j].y * polyKernel(current_test_x, train_data[j].x);
            }
        }
        decision_value += bias;

        double actual_y = test_y(i);
        if ((decision_value >= 0 && actual_y > 0) ||
            (decision_value < 0 && actual_y < 0)) {
            local_correct++;
        }
    }

    int total_correct = 0;
    MPI_Reduce(&local_correct, &total_correct, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double accuracy = 0.0;
    if (rank == 0) {
        if (N_test > 0) {
            accuracy = static_cast<double>(total_correct) / N_test;
        } else {
            accuracy = 0.0;
        }
    }

    MPI_Bcast(&accuracy, 1, MPI_DOUBLE, 0, comm);
    return accuracy;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm comm = MPI_COMM_WORLD;

    std::string train_file = "train_data.csv";
    std::string test_file = "test_data.csv";
    double svm_C = C;
    double main_tol = tol;
    double phase1_tol = 1e-6;
    double phase1_gap_tol = 1e-4;
    int num_rows = -1;
    int test_size = -1;
    int N = 0;
    int d = 0;

    int opt;
    while ((opt = getopt(argc, argv, "r:s:")) != -1) {
        switch (opt) {
            case 'r':
                num_rows = std::stoi(optarg);
                break;
            case 's':
                test_size = std::stoi(optarg);
                break;
            default:
                std::cerr << "Usage: mpirun -np <num_procs> " << argv[0]
                            << " [-r num_train_rows] [-s num_test_rows]"
                            << std::endl;
                MPI_Abort(comm, 1);
                return 1;
        }
    }

    try {
        std::vector<DataPoint> data;
        Eigen::VectorXd y_vec;
        if (world_rank == 0) {
            data = readCSV(train_file, num_rows);
            if(data.empty()) {
                std::cerr << "Error: No training data loaded" << std::endl;
                return 1;
            }
            N = data.size();
            d = data[0].x.size();
            y_vec.resize(N);
            for (int i = 0; i < N; i++) {
                y_vec(i) = data[i].y;
            }
        }
        MPI_Bcast(&N, 1, MPI_INT, 0, comm);
        MPI_Bcast(&d, 1, MPI_INT, 0, comm);

        std::vector<double> flat_data;
        if (world_rank == 0) {
            flat_data.reserve(N * (d + 1));
            for (const auto& point : data) {
                for (int j = 0; j < d; j++) {
                    flat_data.push_back(point.x(j));
                }
                flat_data.push_back(point.y);
            }
        } else {
            flat_data.resize(N * (d + 1));
            data.resize(N);
            y_vec.resize(N);
        }
        
        MPI_Bcast(flat_data.data(), flat_data.size(), MPI_DOUBLE, 0, comm);
        
        if (world_rank != 0) {
            for (int i = 0; i < N; i++) {
                Eigen::VectorXd x_vec(d);
                for (int j = 0; j < d; j++) {
                    x_vec(j) = flat_data[i * (d + 1) + j];
                }
                double y = flat_data[i * (d + 1) + d];
                
                data[i].x = x_vec;
                data[i].y = y;
                y_vec(i) = y;
            }
        }

        Eigen::MatrixXd K = buildKernelMatrix(data, world_rank);
        MPI_Barrier(comm);

        Eigen::MatrixXd Q_svm;
        Eigen::VectorXd c_svm;
        Eigen::MatrixXd A_svm;
        Eigen::VectorXd b_svm;

        if (world_rank == 0) {
            Eigen::MatrixXd Y = y_vec * y_vec.transpose();
            Q_svm = K.array() * Y.array();
            Q_svm += 1e-9 * Eigen::MatrixXd::Identity(N, N);

            c_svm = -Eigen::VectorXd::Ones(N);

            A_svm.resize(1, N);
            A_svm.row(0) = y_vec.transpose();
            b_svm.resize(1);
            b_svm(0) = 0.0;
        } else {
            Q_svm.resize(N,N);
            c_svm.resize(N);
            A_svm.resize(1,N);
            b_svm.resize(1);
        }

        MPI_Barrier(comm);
        double train_start = MPI_Wtime();

        std::optional<Eigen::VectorXd> alpha0_opt = phase_1(
            A_svm, b_svm, svm_C, N,
            phase1_tol,
            t0, mu, barrier_iters, newton_iters,
            phase1_gap_tol,
            comm, world_rank, world_size
        );

        Eigen::VectorXd alpha0 = Eigen::VectorXd::Zero(N);
        if (world_rank == 0 && alpha0_opt) {
            alpha0 = *alpha0_opt;
        }

        Eigen::VectorXd alpha = interiorQP(
            Q_svm, c_svm, A_svm, alpha0,
            svm_C,
            main_tol,
            t0, mu,
            barrier_iters, newton_iters,
            false,
            comm, world_rank, world_size
        );

        MPI_Barrier(comm);
        double train_end = MPI_Wtime();
        double train_time = train_end - train_start;
        double bias = 0.0;

        if (world_rank == 0) {
            bias = computeBias(data, alpha, K);
        }

        double accuracy = 0.0;
        double test_time = 0.0;
        int N_test = 0;

        std::vector<DataPoint> test_data;
        if (world_rank == 0) {
            std::cout << "Loading test data and computing accuracy..." << std::endl;
            test_data = readCSV(test_file, test_size);
            N_test = test_data.size();
        }

        MPI_Bcast(&N_test, 1, MPI_INT, 0, comm);

        if (N_test > 0) {
            MPI_Bcast(alpha.data(), alpha.size(), MPI_DOUBLE, 0, comm);
            MPI_Bcast(&bias, 1, MPI_DOUBLE, 0, comm);

            double test_start = MPI_Wtime();

            accuracy = calculateAccuracy_MPI(data, test_data, alpha, bias,
                                             N, N_test, d,
                                             world_rank, world_size, comm);

            MPI_Barrier(comm);
            double test_end = MPI_Wtime();
            test_time = test_end - test_start;

        } else {
            accuracy = 0.0;
            test_time = 0.0;
        }

        if (world_rank == 0) {
            std::string output_filename = "result_mpi/" + std::to_string(N) + "_" + std::to_string(N_test) + "_" + std::to_string(world_size);
            std::ofstream output_file(output_filename);
            std::ofstream output_file_global("experiment_results.txt", std::ios::app);
                
            if (output_file.is_open()) {
                for (int i = 0; i < alpha.size(); i++) {
                    output_file << alpha[i] << " ";
                }
                output_file << "\n";
                output_file << bias << "\n";
                output_file << train_time << "\n";
                output_file << test_time << "\n";
                output_file << accuracy << "\n";
                output_file.close();
                output_file_global << "MPI: "
                << "Train Rows:" << num_rows
                        << ", Test Rows:" << test_size
                        << ", Train Time:" << train_time
                        << ", Test Time:" << test_time
                        << ", Accuracy:" << accuracy
                        << ", Processes:" << world_size << "\n";
                output_file_global.close();
                std::cout << "MPI results for " << N << " rows, test size " << N_test << ", processes " << world_size << ":" << std::endl;
                std::cout << "Total Training time: " << train_time << " seconds" << std::endl;
                std::cout << "Test time: " << test_time << " seconds" << std::endl;
                std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
                std::cout << "Bias: " << bias << std::endl;
            } else {
                std::cerr << "Error: Unable to open output file: " << output_filename << std::endl;
                MPI_Abort(comm, 1);
            }
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
