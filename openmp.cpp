#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <omp.h>
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
     *
     * Doolittle's method:
     * - L has 1's on the diagonal
     * - U is upper triangular with computed diagonal elements
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
     * First solve Ly = b, then solve Ux = y
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
// PARALLEL LU DECOMPOSITION SOLVER (OpenMP)
// ============================================================================

class ParallelLUSolver {
private:
    int n;
    Matrix L, U;
    int numThreads;
    bool decomposed;

    /**
     * Forward substitution with minimal parallelization
     * Limited by sequential dependencies (y[i] depends on y[0]...y[i-1])
     */
    Vector forwardSubstitution(const Vector& b) {
        Vector y(n, 0.0);
        for (int i = 0; i < n; i++) {
            double sum = 0.0;

            // Parallelize the inner sum for large iterations
#pragma omp parallel for reduction(+:sum) num_threads(numThreads) if(i > 100)
            for (int j = 0; j < i; j++) {
                sum += L[i][j] * y[j];
            }

            y[i] = (b[i] - sum) / L[i][i];
        }
        return y;
    }

    /**
     * Backward substitution with minimal parallelization
     * Limited by sequential dependencies (x[i] depends on x[i+1]...x[n-1])
     */
    Vector backwardSubstitution(const Vector& y) {
        Vector x(n, 0.0);
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;

            // Parallelize the inner sum for large iterations
#pragma omp parallel for reduction(+:sum) num_threads(numThreads) if(n-i > 100)
            for (int j = i + 1; j < n; j++) {
                sum += U[i][j] * x[j];
            }

            if (abs(U[i][i]) < 1e-10) {
#pragma omp critical
                {
                    cerr << "Error: Zero pivot encountered at position " << i << endl;
                }
                exit(1);
            }

            x[i] = (y[i] - sum) / U[i][i];
        }
        return x;
    }

public:
    ParallelLUSolver(int size, int threads) : n(size), numThreads(threads), decomposed(false) {
        L.resize(n, vector<double>(n, 0.0));
        U.resize(n, vector<double>(n, 0.0));
        omp_set_num_threads(numThreads);
    }

    /**
     * Perform LU decomposition with OpenMP parallelization
     * Parallelizes independent computations within each iteration
     * Time Complexity: Still O(n^3) but with parallel speedup
     */
    void decompose(const Matrix& A) {
        // Initialize L and U
#pragma omp parallel for num_threads(numThreads)
        for (int i = 0; i < n; i++) {
            L[i][i] = 1.0;
        }

        // Doolittle's algorithm with OpenMP parallelization
        for (int i = 0; i < n; i++) {
            // Compute U[i][j] for j >= i in parallel
#pragma omp parallel for num_threads(numThreads) schedule(dynamic)
            for (int j = i; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < i; k++) {
                    sum += L[i][k] * U[k][j];
                }
                U[i][j] = A[i][j] - sum;
            }

            // Check for zero pivot
            if (abs(U[i][i]) < 1e-10) {
#pragma omp critical
                {
                    cerr << "Error: Zero pivot encountered at position " << i << endl;
                }
                exit(1);
            }

            // Compute L[j][i] for j > i in parallel
#pragma omp parallel for num_threads(numThreads) schedule(dynamic)
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
// PERFORMANCE MEASUREMENT AND ANALYSIS
// ============================================================================

struct PerformanceMetrics {
    double decompositionTime;
    double substitutionTime;
    double totalTime;
    double speedup;
    double efficiency;
    int threads;
    double solutionError;
    double luError;
};

void runComprehensiveAnalysis(int matrixSize) {
    cout << "\n";
    cout << "+================================================================+\n";
    cout << "|          LU DECOMPOSITION LINEAR EQUATION SOLVER               |\n";
    cout << "|        Serial vs OpenMP Parallel Implementation                |\n";
    cout << "+================================================================+\n";
    cout << "\n";

    cout << "========================================\n";
    cout << "PERFORMANCE ANALYSIS\n";
    cout << "========================================\n";
    cout << "Matrix Size: " << matrixSize << "x" << matrixSize << "\n";
    cout << "Random values with diagonal dominance\n";
    cout << "Thread configurations: 1 (Serial), 2, 4, 8\n";
    cout << "========================================\n\n";

    // Initialize problem with fixed seed for reproducibility
    Matrix A(matrixSize, vector<double>(matrixSize));
    Vector b(matrixSize);

    srand(42); // Fixed seed for reproducibility
    initializeMatrix(A, matrixSize);
    initializeVector(b, matrixSize);

    cout << "Matrix and vector initialized successfully.\n";
    cout << "Diagonal dominance ensured for numerical stability.\n\n";

    vector<PerformanceMetrics> results;
    double serialTime = 0.0;

    // ========== SERIAL EXECUTION ==========
    cout << "+-------------------------------------+\n";
    cout << "|      SERIAL VERSION (1 Thread)      |\n";
    cout << "+-------------------------------------+\n";
    {
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

        PerformanceMetrics metric;
        metric.decompositionTime = decompTime;
        metric.substitutionTime = substTime;
        metric.totalTime = serialTime;
        metric.speedup = 1.0;
        metric.efficiency = 1.0;
        metric.threads = 1;
        metric.solutionError = solutionError;
        metric.luError = luError;
        results.push_back(metric);

        cout << "  Decomposition Time: " << fixed << setprecision(6) << decompTime << " seconds\n";
        cout << "  Substitution Time:  " << fixed << setprecision(6) << substTime << " seconds\n";
        cout << "  Total Time:         " << fixed << setprecision(6) << serialTime << " seconds\n";
        cout << "  Solution Error:     " << scientific << setprecision(4) << solutionError << "\n";
        cout << "  LU Reconstruction:  " << scientific << setprecision(4) << luError << "\n";
        cout << fixed << "\n";
    }

    // ========== PARALLEL EXECUTION ==========
    vector<int> threadCounts = { 2, 4, 8 };

    for (int threads : threadCounts) {
        cout << "+-------------------------------------+\n";
        cout << "|   PARALLEL VERSION (" << threads << " Threads)    |\n";
        cout << "+-------------------------------------+\n";

        ParallelLUSolver solver(matrixSize, threads);

        auto start = chrono::high_resolution_clock::now();
        solver.decompose(A);
        auto decompEnd = chrono::high_resolution_clock::now();
        Vector x = solver.solve(b);
        auto end = chrono::high_resolution_clock::now();

        double decompTime = chrono::duration<double>(decompEnd - start).count();
        double substTime = chrono::duration<double>(end - decompEnd).count();
        double parallelTime = chrono::duration<double>(end - start).count();
        double speedup = serialTime / parallelTime;
        double efficiency = speedup / threads;

        // Verify solution
        double solutionError = verifySolution(A, x, b);
        double luError = verifyLUDecomposition(A, solver.getL(), solver.getU());

        PerformanceMetrics metric;
        metric.decompositionTime = decompTime;
        metric.substitutionTime = substTime;
        metric.totalTime = parallelTime;
        metric.speedup = speedup;
        metric.efficiency = efficiency;
        metric.threads = threads;
        metric.solutionError = solutionError;
        metric.luError = luError;
        results.push_back(metric);

        cout << "  Decomposition Time: " << fixed << setprecision(6) << decompTime << " seconds\n";
        cout << "  Substitution Time:  " << fixed << setprecision(6) << substTime << " seconds\n";
        cout << "  Total Time:         " << fixed << setprecision(6) << parallelTime << " seconds\n";
        cout << "  Speedup:            " << fixed << setprecision(4) << speedup << "x\n";
        cout << "  Efficiency:         " << fixed << setprecision(2) << (efficiency * 100) << "%\n";
        cout << "  Solution Error:     " << scientific << setprecision(4) << solutionError << "\n";
        cout << "  LU Reconstruction:  " << scientific << setprecision(4) << luError << "\n";
        cout << fixed << "\n";
    }

    // ========== DETAILED SUMMARY TABLE ==========
    cout << "\n+================================================================================+\n";
    cout << "|                           PERFORMANCE SUMMARY                                  |\n";
    cout << "+================================================================================+\n\n";

    cout << left
        << setw(12) << "Version"
        << setw(10) << "Threads"
        << setw(16) << "Decomp (s)"
        << setw(16) << "Subst (s)"
        << setw(16) << "Total (s)"
        << setw(12) << "Speedup"
        << setw(14) << "Efficiency(%)\n";
    cout << string(96, '-') << "\n";

    for (const auto& metric : results) {
        string version = (metric.threads == 1) ? "Serial" : "Parallel";
        cout << left
            << setw(12) << version
            << setw(10) << metric.threads
            << setw(16) << fixed << setprecision(6) << metric.decompositionTime
            << setw(16) << fixed << setprecision(6) << metric.substitutionTime
            << setw(16) << fixed << setprecision(6) << metric.totalTime
            << setw(12) << fixed << setprecision(4) << metric.speedup
            << setw(14) << fixed << setprecision(2) << (metric.efficiency * 100) << "\n";
    }
    cout << string(96, '-') << "\n\n";

    // ========== CORRECTNESS VERIFICATION ==========
    cout << "+================================================================================+\n";
    cout << "|                        CORRECTNESS VERIFICATION                                |\n";
    cout << "+================================================================================+\n\n";

    cout << "All implementations verified for numerical accuracy:\n";
    cout << "- Solution Error (||Ax - b||): Maximum residual across all methods\n";
    cout << "- LU Reconstruction Error (||LU - A||): Decomposition accuracy\n\n";

    bool allCorrect = true;
    for (const auto& metric : results) {
        string version = (metric.threads == 1) ? "Serial" : "Parallel (" + to_string(metric.threads) + " threads)";
        cout << version << ":\n";
        cout << "  Solution Error:        " << scientific << setprecision(4) << metric.solutionError;
        if (metric.solutionError < 1e-6) {
            cout << " [PASS]\n";
        }
        else {
            cout << " [FAIL]\n";
            allCorrect = false;
        }
        cout << "  LU Reconstruction:     " << scientific << setprecision(4) << metric.luError;
        if (metric.luError < 1e-6) {
            cout << " [PASS]\n";
        }
        else {
            cout << " [FAIL]\n";
            allCorrect = false;
        }
        cout << "\n";
    }

    if (allCorrect) {
        cout << "[+] All implementations produce correct results!\n\n";
    }
    else {
        cout << "[-] Some implementations have numerical issues!\n\n";
    }

    // ========== PERFORMANCE ANALYSIS ==========
    cout << "+================================================================================+\n";
    cout << "|                          PERFORMANCE INSIGHTS                                  |\n";
    cout << "+================================================================================+\n\n";

    cout << "Key Observations:\n\n";

    cout << "1. SPEEDUP ANALYSIS:\n";
    cout << "   - 2 threads: " << fixed << setprecision(3) << results[1].speedup << "x speedup\n";
    cout << "   - 4 threads: " << fixed << setprecision(3) << results[2].speedup << "x speedup\n";
    cout << "   - 8 threads: " << fixed << setprecision(3) << results[3].speedup << "x speedup\n";
    cout << "   " << (results[3].speedup > results[2].speedup ? "[+] Speedup increases with thread count"
        : "[!] Diminishing returns observed") << "\n\n";

    cout << "2. PARALLEL EFFICIENCY:\n";
    for (size_t i = 1; i < results.size(); i++) {
        cout << "   - " << results[i].threads << " threads: "
            << fixed << setprecision(1) << (results[i].efficiency * 100) << "%";
        if (results[i].efficiency > 0.7) {
            cout << " (Excellent)\n";
        }
        else if (results[i].efficiency > 0.5) {
            cout << " (Good)\n";
        }
        else if (results[i].efficiency > 0.3) {
            cout << " (Fair)\n";
        }
        else {
            cout << " (Poor - overhead dominates)\n";
        }
    }
    cout << "\n";

    cout << "================================================================================\n";
    cout << "                             ANALYSIS COMPLETE                                  \n";
    cout << "================================================================================\n\n";
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    // Set up OpenMP
    cout << "OpenMP Configuration:\n";
    cout << "  Max threads available: " << omp_get_max_threads() << "\n";

    // Run comprehensive analysis with 500x500 matrix
    const int MATRIX_SIZE = 500;
    runComprehensiveAnalysis(MATRIX_SIZE);

    return 0;
}