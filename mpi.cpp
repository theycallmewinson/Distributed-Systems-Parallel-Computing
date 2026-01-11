#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <string>

using namespace std;
using Matrix = vector<vector<double>>;
using Vector = vector<double>;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Initialize a random matrix with diagonal dominance
 * Diagonal dominance ensures the matrix is non-singular and solvable
 */
void initializeMatrix(Matrix& A, int n) {
    for (int i = 0; i < n; i++) {
        double rowSum = 0.0;
        for (int j = 0; j < n; j++) {
            A[i][j] = (rand() % 100 + 1) / 10.0; // Random values 0.1 to 10.0
            if (i != j) {
                rowSum += abs(A[i][j]);
            }
        }
        // Make diagonally dominant: |A[i][i]| > sum of |A[i][j]| for j != i
        A[i][i] = rowSum + (rand() % 50 + 50); // Ensure diagonal dominance
    }
}

/**
 * Initialize a random vector
 */
void initializeVector(Vector& b, int n) {
    for (int i = 0; i < n; i++) {
        b[i] = (rand() % 100 + 1) / 10.0;
    }
}

/**
 * Print matrix (for debugging small matrices)
 */
void printMatrix(const Matrix& A, const string& name, int maxSize = 10) {
    if (A.size() > maxSize) {
        cout << name << ": [Matrix too large to display, size: "
            << A.size() << "x" << A[0].size() << "]\n";
        return;
    }

    cout << name << ":\n";
    for (const auto& row : A) {
        for (double val : row) {
            cout << setw(10) << fixed << setprecision(4) << val << " ";
        }
        cout << "\n";
    }
    cout << "\n";
}

/**
 * Print vector (for debugging)
 */
void printVector(const Vector& v, const string& name, int maxSize = 10) {
    cout << name << ": [";
    int displaySize = min((int)v.size(), maxSize);
    for (int i = 0; i < displaySize; i++) {
        cout << fixed << setprecision(4) << v[i];
        if (i < displaySize - 1) cout << ", ";
    }
    if (v.size() > maxSize) {
        cout << " ... (" << v.size() << " elements)";
    }
    cout << "]\n";
}

/**
 * Verify solution: Check if Ax = b
 * Returns the maximum residual error
 */
double verifySolution(const Matrix& A, const Vector& x, const Vector& b) {
    int n = A.size();
    double maxError = 0.0;

    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        double error = abs(sum - b[i]);
        maxError = max(maxError, error);
    }

    return maxError;
}

/**
 * Verify LU decomposition: Check if L*U = A
 * Returns the maximum reconstruction error
 */
double verifyLUDecomposition(const Matrix& A, const Matrix& L, const Matrix& U) {
    int n = A.size();
    double maxError = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += L[i][k] * U[k][j];
            }
            double error = abs(sum - A[i][j]);
            maxError = max(maxError, error);
        }
    }

    return maxError;
}

// ============================================================================
// SERIAL LU DECOMPOSITION SOLVER
// ============================================================================

class SerialLUSolver {
private:
    int n;
    Matrix L, U;
    bool decomposed;

    /**
     * Forward substitution: Solve Ly = b
     * Time Complexity: O(n^2)
     */
    Vector forwardSubstitution(const Vector& b) {
        Vector y(n, 0.0);
        for (int i = 0; i < n; i++) {
            y[i] = b[i];
            for (int j = 0; j < i; j++) {
                y[i] -= L[i][j] * y[j];
            }
            y[i] /= L[i][i];
        }
        return y;
    }

    /**
     * Backward substitution: Solve Ux = y
     * Time Complexity: O(n^2)
     */
    Vector backwardSubstitution(const Vector& y) {
        Vector x(n, 0.0);
        for (int i = n - 1; i >= 0; i--) {
            x[i] = y[i];
            for (int j = i + 1; j < n; j++) {
                x[i] -= U[i][j] * x[j];
            }

            if (abs(U[i][i]) < 1e-10) {
                cerr << "Error: Zero pivot encountered at position " << i << endl;
                exit(1);
            }

            x[i] /= U[i][i];
        }
        return x;
    }

public:
    SerialLUSolver(int size) : n(size), decomposed(false) {
        L.resize(n, vector<double>(n, 0.0));
        U.resize(n, vector<double>(n, 0.0));
    }

    /**
     * Perform LU decomposition using Doolittle's algorithm
     * Time Complexity: O(n^3)
     */
    void decompose(const Matrix& A) {
        // Initialize L as identity matrix and U as zero
        for (int i = 0; i < n; i++) {
            L[i][i] = 1.0;
        }

        // Doolittle's algorithm
        for (int i = 0; i < n; i++) {
            // Compute U[i][j] for j >= i
            for (int j = i; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < i; k++) {
                    sum += L[i][k] * U[k][j];
                }
                U[i][j] = A[i][j] - sum;
            }

            // Check for zero pivot
            if (abs(U[i][i]) < 1e-10) {
                cerr << "Error: Zero pivot encountered at position " << i << endl;
                exit(1);
            }

            // Compute L[j][i] for j > i
            for (int j = i + 1; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < i; k++) {
                    sum += L[j][k] * U[k][i];
                }
                L[j][i] = (A[j][i] - sum) / U[i][i];
            }
        }

        decomposed = true;
    }

    /**
     * Solve Ax = b using LU decomposition
     */
    Vector solve(const Vector& b) {
        if (!decomposed) {
            cerr << "Error: Must decompose matrix before solving!" << endl;
            exit(1);
        }

        Vector y = forwardSubstitution(b);
        Vector x = backwardSubstitution(y);
        return x;
    }

    const Matrix& getL() const { return L; }
    const Matrix& getU() const { return U; }
};

// ============================================================================
// PARALLEL LU DECOMPOSITION SOLVER (MPI)
// ============================================================================

class MPILUSolver {
private:
    int n;
    Matrix L, U;
    int rank, size;
    bool decomposed;

    /**
     * Forward substitution with MPI parallelization
     * Each process computes partial sums for its assigned rows
     */
    Vector forwardSubstitution(const Vector& b) {
        Vector y(n, 0.0);

        for (int i = 0; i < n; i++) {
            // Broadcast the previously computed y[i-1] values needed
            if (i > 0) {
                MPI_Bcast(y.data(), i, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }

            if (rank == 0) {
                double sum = 0.0;
                for (int j = 0; j < i; j++) {
                    sum += L[i][j] * y[j];
                }
                y[i] = (b[i] - sum) / L[i][i];
            }
        }

        // Broadcast final result
        MPI_Bcast(y.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        return y;
    }

    /**
     * Backward substitution with MPI parallelization
     */
    Vector backwardSubstitution(const Vector& y) {
        Vector x(n, 0.0);

        for (int i = n - 1; i >= 0; i--) {
            // Broadcast the previously computed x[i+1] values needed
            if (i < n - 1) {
                MPI_Bcast(x.data() + i + 1, n - i - 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }

            if (rank == 0) {
                double sum = 0.0;
                for (int j = i + 1; j < n; j++) {
                    sum += U[i][j] * x[j];
                }

                if (abs(U[i][i]) < 1e-10) {
                    cerr << "Error: Zero pivot encountered at position " << i << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                x[i] = (y[i] - sum) / U[i][i];
            }
        }

        // Broadcast final result
        MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        return x;
    }

public:
    MPILUSolver(int size_n, int mpi_rank, int mpi_size)
        : n(size_n), rank(mpi_rank), size(mpi_size), decomposed(false) {
        L.resize(n, vector<double>(n, 0.0));
        U.resize(n, vector<double>(n, 0.0));
    }

    /**
     * Perform LU decomposition with MPI parallelization
     * Distributes row computations across processes
     */
    void decompose(const Matrix& A) {
        // Initialize L as identity matrix
        for (int i = 0; i < n; i++) {
            L[i][i] = 1.0;
        }

        // Doolittle's algorithm with MPI parallelization
        for (int i = 0; i < n; i++) {
            // Compute U[i][j] for j >= i - distribute columns across processes
            for (int j = i + rank; j < n; j += size) {
                double sum = 0.0;
                for (int k = 0; k < i; k++) {
                    sum += L[i][k] * U[k][j];
                }
                U[i][j] = A[i][j] - sum;
            }

            // Gather U row i from all processes
            for (int j = i; j < n; j++) {
                int owner = (j - i) % size;
                MPI_Bcast(&U[i][j], 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
            }

            // Check for zero pivot on root
            if (rank == 0 && abs(U[i][i]) < 1e-10) {
                cerr << "Error: Zero pivot encountered at position " << i << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Broadcast pivot value
            MPI_Bcast(&U[i][i], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Compute L[j][i] for j > i - distribute rows across processes
            for (int j = i + 1 + rank; j < n; j += size) {
                double sum = 0.0;
                for (int k = 0; k < i; k++) {
                    sum += L[j][k] * U[k][i];
                }
                L[j][i] = (A[j][i] - sum) / U[i][i];
            }

            // Gather L column i from all processes
            for (int j = i + 1; j < n; j++) {
                int owner = (j - i - 1) % size;
                MPI_Bcast(&L[j][i], 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
            }
        }

        decomposed = true;
    }

    /**
     * Solve Ax = b using LU decomposition
     */
    Vector solve(const Vector& b) {
        if (!decomposed) {
            if (rank == 0) {
                cerr << "Error: Must decompose matrix before solving!" << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        Vector y = forwardSubstitution(b);
        Vector x = backwardSubstitution(y);
        return x;
    }

    const Matrix& getL() const { return L; }
    const Matrix& getU() const { return U; }
};

// ============================================================================
// PERFORMANCE MEASUREMENT AND ANALYSIS
// ============================================================================

struct PerformanceMetrics {
    double decompositionTime;
    double substitutionTime;
    double totalTime;
    double speedup;
    double efficiency;
    int processes;
    double solutionError;
    double luError;
};

void runComprehensiveAnalysis(int matrixSize, int rank, int size) {
    if (rank == 0) {
        cout << "\n";
        cout << "+================================================================+\n";
        cout << "|          LU DECOMPOSITION LINEAR EQUATION SOLVER               |\n";
        cout << "|           Serial vs MPI Parallel Implementation                |\n";
        cout << "+================================================================+\n";
        cout << "\n";

        cout << "========================================\n";
        cout << "PERFORMANCE ANALYSIS\n";
        cout << "========================================\n";
        cout << "Matrix Size: " << matrixSize << "x" << matrixSize << "\n";
        cout << "Random values with diagonal dominance\n";
        cout << "MPI Processes: " << size << "\n";
        cout << "========================================\n\n";
    }

    // Initialize problem with fixed seed for reproducibility
    Matrix A(matrixSize, vector<double>(matrixSize));
    Vector b(matrixSize);

    if (rank == 0) {
        srand(42); // Fixed seed for reproducibility
        initializeMatrix(A, matrixSize);
        initializeVector(b, matrixSize);
        cout << "Matrix and vector initialized successfully.\n";
        cout << "Diagonal dominance ensured for numerical stability.\n\n";
    }

    // Broadcast matrix and vector to all processes
    for (int i = 0; i < matrixSize; i++) {
        MPI_Bcast(A[i].data(), matrixSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(b.data(), matrixSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double serialTime = 0.0;
    PerformanceMetrics serialMetric, parallelMetric;

    // ========== SERIAL EXECUTION (only on rank 0) ==========
    if (rank == 0) {
        cout << "+-------------------------------------+\n";
        cout << "|      SERIAL VERSION (1 Process)     |\n";
        cout << "+-------------------------------------+\n";

        SerialLUSolver solver(matrixSize);

        auto start = chrono::high_resolution_clock::now();
        solver.decompose(A);
        auto decompEnd = chrono::high_resolution_clock::now();
        Vector x = solver.solve(b);
        auto end = chrono::high_resolution_clock::now();

        double decompTime = chrono::duration<double>(decompEnd - start).count();
        double substTime = chrono::duration<double>(end - decompEnd).count();
        serialTime = chrono::duration<double>(end - start).count();

        // Verify solution
        double solutionError = verifySolution(A, x, b);
        double luError = verifyLUDecomposition(A, solver.getL(), solver.getU());

        serialMetric.decompositionTime = decompTime;
        serialMetric.substitutionTime = substTime;
        serialMetric.totalTime = serialTime;
        serialMetric.speedup = 1.0;
        serialMetric.efficiency = 1.0;
        serialMetric.processes = 1;
        serialMetric.solutionError = solutionError;
        serialMetric.luError = luError;

        cout << "  Decomposition Time: " << fixed << setprecision(6) << decompTime << " seconds\n";
        cout << "  Substitution Time:  " << fixed << setprecision(6) << substTime << " seconds\n";
        cout << "  Total Time:         " << fixed << setprecision(6) << serialTime << " seconds\n";
        cout << "  Solution Error:     " << scientific << setprecision(4) << solutionError << "\n";
        cout << "  LU Reconstruction:  " << scientific << setprecision(4) << luError << "\n";
        cout << fixed << "\n";
    }

    // Broadcast serial time to all processes
    MPI_Bcast(&serialTime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ========== PARALLEL EXECUTION (MPI) ==========
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "+-------------------------------------+\n";
        cout << "|  MPI PARALLEL VERSION (" << size << " Processes)|\n";
        cout << "+-------------------------------------+\n";
    }

    MPILUSolver solver(matrixSize, rank, size);

    double start = MPI_Wtime();
    solver.decompose(A);
    double decompEnd = MPI_Wtime();
    Vector x = solver.solve(b);
    double end = MPI_Wtime();

    double decompTime = decompEnd - start;
    double substTime = end - decompEnd;
    double parallelTime = end - start;

    // Gather timing from all processes and use max
    double maxDecompTime, maxSubstTime, maxParallelTime;
    MPI_Reduce(&decompTime, &maxDecompTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&substTime, &maxSubstTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&parallelTime, &maxParallelTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double speedup = serialTime / maxParallelTime;
        double efficiency = speedup / size;

        // Verify solution
        double solutionError = verifySolution(A, x, b);
        double luError = verifyLUDecomposition(A, solver.getL(), solver.getU());

        parallelMetric.decompositionTime = maxDecompTime;
        parallelMetric.substitutionTime = maxSubstTime;
        parallelMetric.totalTime = maxParallelTime;
        parallelMetric.speedup = speedup;
        parallelMetric.efficiency = efficiency;
        parallelMetric.processes = size;
        parallelMetric.solutionError = solutionError;
        parallelMetric.luError = luError;

        cout << "  Decomposition Time: " << fixed << setprecision(6) << maxDecompTime << " seconds\n";
        cout << "  Substitution Time:  " << fixed << setprecision(6) << maxSubstTime << " seconds\n";
        cout << "  Total Time:         " << fixed << setprecision(6) << maxParallelTime << " seconds\n";
        cout << "  Speedup:            " << fixed << setprecision(4) << speedup << "x\n";
        cout << "  Efficiency:         " << fixed << setprecision(2) << (efficiency * 100) << "%\n";
        cout << "  Solution Error:     " << scientific << setprecision(4) << solutionError << "\n";
        cout << "  LU Reconstruction:  " << scientific << setprecision(4) << luError << "\n";
        cout << fixed << "\n";

        // ========== PERFORMANCE SUMMARY ==========
        cout << "\n+================================================================================+\n";
        cout << "|                           PERFORMANCE SUMMARY                                  |\n";
        cout << "+================================================================================+\n\n";

        cout << left
            << setw(12) << "Version"
            << setw(12) << "Processes"
            << setw(16) << "Decomp (s)"
            << setw(16) << "Subst (s)"
            << setw(16) << "Total (s)"
            << setw(12) << "Speedup"
            << setw(14) << "Efficiency(%)\n";
        cout << string(98, '-') << "\n";

        cout << left
            << setw(12) << "Serial"
            << setw(12) << serialMetric.processes
            << setw(16) << fixed << setprecision(6) << serialMetric.decompositionTime
            << setw(16) << fixed << setprecision(6) << serialMetric.substitutionTime
            << setw(16) << fixed << setprecision(6) << serialMetric.totalTime
            << setw(12) << fixed << setprecision(4) << serialMetric.speedup
            << setw(14) << fixed << setprecision(2) << (serialMetric.efficiency * 100) << "\n";

        cout << left
            << setw(12) << "MPI"
            << setw(12) << parallelMetric.processes
            << setw(16) << fixed << setprecision(6) << parallelMetric.decompositionTime
            << setw(16) << fixed << setprecision(6) << parallelMetric.substitutionTime
            << setw(16) << fixed << setprecision(6) << parallelMetric.totalTime
            << setw(12) << fixed << setprecision(4) << parallelMetric.speedup
            << setw(14) << fixed << setprecision(2) << (parallelMetric.efficiency * 100) << "\n";

        cout << string(98, '-') << "\n\n";

        // ========== CORRECTNESS VERIFICATION ==========
        cout << "+================================================================================+\n";
        cout << "|                        CORRECTNESS VERIFICATION                                |\n";
        cout << "+================================================================================+\n\n";

        cout << "Serial Version:\n";
        cout << "  Solution Error:        " << scientific << setprecision(4) << serialMetric.solutionError;
        cout << (serialMetric.solutionError < 1e-6 ? " [PASS]\n" : " [FAIL]\n");
        cout << "  LU Reconstruction:     " << scientific << setprecision(4) << serialMetric.luError;
        cout << (serialMetric.luError < 1e-6 ? " [PASS]\n\n" : " [FAIL]\n\n");

        cout << "MPI Parallel Version (" << size << " processes):\n";
        cout << "  Solution Error:        " << scientific << setprecision(4) << parallelMetric.solutionError;
        cout << (parallelMetric.solutionError < 1e-6 ? " [PASS]\n" : " [FAIL]\n");
        cout << "  LU Reconstruction:     " << scientific << setprecision(4) << parallelMetric.luError;
        cout << (parallelMetric.luError < 1e-6 ? " [PASS]\n\n" : " [FAIL]\n\n");

        bool allCorrect = (serialMetric.solutionError < 1e-6 && serialMetric.luError < 1e-6 &&
            parallelMetric.solutionError < 1e-6 && parallelMetric.luError < 1e-6);

        cout << (allCorrect ? "[+] All implementations produce correct results!\n\n"
            : "[-] Some implementations have numerical issues!\n\n");

        // ========== PERFORMANCE INSIGHTS ==========
        cout << "+================================================================================+\n";
        cout << "|                          PERFORMANCE INSIGHTS                                  |\n";
        cout << "+================================================================================+\n\n";

        cout << "Key Observations:\n\n";

        cout << "1. MPI SPEEDUP ANALYSIS:\n";
        cout << "   - " << size << " processes: " << fixed << setprecision(3) << speedup << "x speedup\n";
        cout << "   - Parallel efficiency: " << fixed << setprecision(1) << (efficiency * 100) << "%\n\n";

        cout << "2. COMMUNICATION OVERHEAD:\n";
        cout << "   - MPI involves inter-process communication via MPI_Bcast\n";
        cout << "   - Each iteration requires synchronization across processes\n";
        cout << "   - Network latency can impact performance in distributed systems\n\n";

        cout << "3. DECOMPOSITION vs SUBSTITUTION:\n";
        cout << "   - Decomposition is O(n³) and parallelizes well across processes\n";
        cout << "   - Substitution has sequential dependencies limiting parallelization\n";
        cout << "   - Row/column distribution enables effective load balancing\n\n";

        cout << "4. MPI SCALABILITY:\n";
        cout << "   - Row-wise distribution of computation across processes\n";
        cout << "   - Broadcast operations ensure data consistency\n";
        cout << "   - Suitable for distributed memory systems and clusters\n";
        cout << "   - Scales better with larger matrices due to computation/communication ratio\n\n";

        cout << "5. COMPARISON WITH OPENMP:\n";
        cout << "   - MPI: Distributed memory, works across multiple nodes\n";
        cout << "   - OpenMP: Shared memory, limited to single node\n";
        cout << "   - MPI has higher communication overhead but better scalability\n";
        cout << "   - Choose based on hardware: OpenMP for single node, MPI for clusters\n\n";

        cout << "================================================================================\n";
        cout << "                             ANALYSIS COMPLETE                                  \n";
        cout << "================================================================================\n\n";
    }
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        cout << "MPI Configuration:\n";
        cout << "  Total processes: " << size << "\n";
        cout << "  MPI initialized successfully\n\n";
    }

    // Run comprehensive analysis with 500x500 matrix
    const int MATRIX_SIZE = 500;
    runComprehensiveAnalysis(MATRIX_SIZE, rank, size);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}