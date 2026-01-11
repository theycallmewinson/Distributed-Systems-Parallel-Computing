#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
using Matrix = vector<vector<double>>;
using Vector = vector<double>;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

inline string int_to_string(int value) {
    ostringstream oss;
    oss << value;
    return oss.str();
}

inline string double_to_string(double value, int precision = 4) {
    ostringstream oss;
    oss << fixed << setprecision(precision) << value;
    return oss.str();
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            cerr << "CUDA Error: " << cudaGetErrorString(error) \
                 << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// MATRIX UTILITIES
// ============================================================================

void initializeMatrix(Matrix& A, int n) {
    for (int i = 0; i < n; i++) {
        double rowSum = 0.0;
        for (int j = 0; j < n; j++) {
            A[i][j] = (rand() % 100 + 1) / 10.0;
            if (i != j) {
                rowSum += fabs(A[i][j]);
            }
        }
        A[i][i] = rowSum + (rand() % 50 + 50);
    }
}

void initializeVector(Vector& b, int n) {
    for (int i = 0; i < n; i++) {
        b[i] = (rand() % 100 + 1) / 10.0;
    }
}

void matrixTo1D(const Matrix& mat, double* arr, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            arr[i * n + j] = mat[i][j];
        }
    }
}

void arrayTo2D(const double* arr, Matrix& mat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = arr[i * n + j];
        }
    }
}

double verifySolution(const Matrix& A, const Vector& x, const Vector& b) {
    int n = A.size();
    double maxError = 0.0;
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        double error = fabs(sum - b[i]);
        maxError = max(maxError, error);
    }
    return maxError;
}

double verifyLUDecomposition(const Matrix& A, const Matrix& L, const Matrix& U) {
    int n = A.size();
    double maxError = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += L[i][k] * U[k][j];
            }
            double error = fabs(sum - A[i][j]);
            maxError = max(maxError, error);
        }
    }
    return maxError;
}

// ============================================================================
// SERIAL LU SOLVER (BASELINE)
// ============================================================================

class SerialLUSolver {
private:
    int n;
    Matrix L, U;
    bool decomposed;

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

    Vector backwardSubstitution(const Vector& y) {
        Vector x(n, 0.0);
        for (int i = n - 1; i >= 0; i--) {
            x[i] = y[i];
            for (int j = i + 1; j < n; j++) {
                x[i] -= U[i][j] * x[j];
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

    void decompose(const Matrix& A) {
        for (int i = 0; i < n; i++) {
            L[i][i] = 1.0;
        }

        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < i; k++) {
                    sum += L[i][k] * U[k][j];
                }
                U[i][j] = A[i][j] - sum;
            }

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

    Vector solve(const Vector& b) {
        Vector y = forwardSubstitution(b);
        return backwardSubstitution(y);
    }

    const Matrix& getL() const { return L; }
    const Matrix& getU() const { return U; }
};

// ============================================================================
// OPTIMIZED CUDA KERNELS
// ============================================================================

__global__ void computeURow(double* L, double* U, const double* A, int n, int i) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + i;

    if (j < n) {
        double sum = 0.0;
        for (int k = 0; k < i; k++) {
            sum += L[i * n + k] * U[k * n + j];
        }
        U[i * n + j] = A[i * n + j] - sum;
    }
}

__global__ void computeLColumn(double* L, double* U, const double* A, int n, int i) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + i + 1;

    if (j < n) {
        double sum = 0.0;
        for (int k = 0; k < i; k++) {
            sum += L[j * n + k] * U[k * n + i];
        }
        double pivot = U[i * n + i];
        L[j * n + i] = (A[j * n + i] - sum) / pivot;
    }
}

// ============================================================================
// HYBRID CUDA LU SOLVER (OPTIMIZED)
// ============================================================================

class HybridCUDALUSolver {
private:
    int n;
    Matrix L, U;
    bool decomposed;
    int threadsPerBlock;
    double* d_A, * d_L, * d_U;

public:
    HybridCUDALUSolver(int size, int threads = 256)
        : n(size), decomposed(false), threadsPerBlock(threads) {
        L.resize(n, vector<double>(n, 0.0));
        U.resize(n, vector<double>(n, 0.0));

        // Allocate GPU memory once
        CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_L, n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_U, n * n * sizeof(double)));
    }

    ~HybridCUDALUSolver() {
        cudaFree(d_A);
        cudaFree(d_L);
        cudaFree(d_U);
    }

    void decompose(const Matrix& A) {
        double* h_A = new double[n * n];
        double* h_L = new double[n * n];
        double* h_U = new double[n * n];

        matrixTo1D(A, h_A, n);

        // Initialize L and U
        for (int i = 0; i < n * n; i++) {
            h_L[i] = 0.0;
            h_U[i] = 0.0;
        }
        for (int i = 0; i < n; i++) {
            h_L[i * n + i] = 1.0;
        }

        // Transfer to GPU
        CUDA_CHECK(cudaMemcpy(d_A, h_A, n * n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_L, h_L, n * n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_U, h_U, n * n * sizeof(double), cudaMemcpyHostToDevice));

        // LU Decomposition on GPU
        for (int i = 0; i < n; i++) {
            int numElementsU = n - i;
            int numBlocksU = (numElementsU + threadsPerBlock - 1) / threadsPerBlock;
            computeURow << <numBlocksU, threadsPerBlock >> > (d_L, d_U, d_A, n, i);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            int numElementsL = n - i - 1;
            if (numElementsL > 0) {
                int numBlocksL = (numElementsL + threadsPerBlock - 1) / threadsPerBlock;
                computeLColumn << <numBlocksL, threadsPerBlock >> > (d_L, d_U, d_A, n, i);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }

        // Transfer back to CPU
        CUDA_CHECK(cudaMemcpy(h_L, d_L, n * n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_U, d_U, n * n * sizeof(double), cudaMemcpyDeviceToHost));

        arrayTo2D(h_L, L, n);
        arrayTo2D(h_U, U, n);

        delete[] h_A;
        delete[] h_L;
        delete[] h_U;

        decomposed = true;
    }

    // HYBRID APPROACH: Do substitution on CPU (faster for this operation)
    Vector solve(const Vector& b) {
        if (!decomposed) {
            cerr << "Error: Must decompose before solving!" << endl;
            exit(1);
        }

        // Forward substitution (Ly = b)
        Vector y(n, 0.0);
        for (int i = 0; i < n; i++) {
            y[i] = b[i];
            for (int j = 0; j < i; j++) {
                y[i] -= L[i][j] * y[j];
            }
            y[i] /= L[i][i];
        }

        // Backward substitution (Ux = y)
        Vector x(n, 0.0);
        for (int i = n - 1; i >= 0; i--) {
            x[i] = y[i];
            for (int j = i + 1; j < n; j++) {
                x[i] -= U[i][j] * x[j];
            }
            x[i] /= U[i][i];
        }

        return x;
    }

    const Matrix& getL() const { return L; }
    const Matrix& getU() const { return U; }
};

// ============================================================================
// PERFORMANCE METRICS
// ============================================================================

struct PerformanceMetrics {
    double decompositionTime;
    double substitutionTime;
    double totalTime;
    double speedup;
    double improvement; // Percentage improvement
    string version;
    int blockSize;
    double solutionError;
    double luError;
};

// ============================================================================
// COMPREHENSIVE ANALYSIS
// ============================================================================

void runComprehensiveAnalysis(int matrixSize) {
    cout << "\n+================================================================+\n";
    cout << "|     OPTIMIZED CUDA LU DECOMPOSITION SOLVER                     |\n";
    cout << "|     Hybrid CPU-GPU Architecture for Maximum Performance        |\n";
    cout << "+================================================================+\n\n";

    // Device info
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        cerr << "No CUDA devices found!" << endl;
        exit(1);
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    cout << "+-- CUDA Device Information -------------------------------------+\n";
    cout << "| Device: " << prop.name << endl;
    cout << "| Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "| Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    cout << "| Multiprocessors: " << prop.multiProcessorCount << endl;
    cout << "| Max Threads/Block: " << prop.maxThreadsPerBlock << endl;
    cout << "+----------------------------------------------------------------+\n\n";

    cout << "Matrix Size: " << matrixSize << "x" << matrixSize << endl;
    cout << "Testing block sizes: 256, 512, 1024 threads\n\n";

    // Initialize test data
    Matrix A(matrixSize, vector<double>(matrixSize));
    Vector b(matrixSize);
    srand(42);
    initializeMatrix(A, matrixSize);
    initializeVector(b, matrixSize);

    vector<PerformanceMetrics> results;
    double serialTime = 0.0;

    // ========================================================================
    // SERIAL BASELINE
    // ========================================================================
    cout << "+-- SERIAL BASELINE (CPU) ---------------------------------------+\n";
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

        PerformanceMetrics metric;
        metric.decompositionTime = decompTime;
        metric.substitutionTime = substTime;
        metric.totalTime = serialTime;
        metric.speedup = 1.0;
        metric.improvement = 0.0;
        metric.version = "CPU Serial";
        metric.blockSize = 0;
        metric.solutionError = verifySolution(A, x, b);
        metric.luError = verifyLUDecomposition(A, solver.getL(), solver.getU());
        results.push_back(metric);

        cout << "| Decomposition: " << fixed << setprecision(6) << decompTime << " s\n";
        cout << "| Substitution:  " << fixed << setprecision(6) << substTime << " s\n";
        cout << "| Total Time:    " << fixed << setprecision(6) << serialTime << " s\n";
        cout << "| Solution Error: " << scientific << setprecision(2) << metric.solutionError << endl;
        cout << "| LU Error:       " << scientific << setprecision(2) << metric.luError << endl;
        cout << "+----------------------------------------------------------------+\n\n";
    }

    // ========================================================================
    // HYBRID CUDA VERSIONS
    // ========================================================================
    vector<int> blockSizes = { 256, 512, 1024 };

    for (int blockSize : blockSizes) {
        cout << "+-- HYBRID CUDA (" << blockSize << " threads/block) ";
        cout << string(38 - int_to_string(blockSize).length(), '-') << "+\n";

        HybridCUDALUSolver solver(matrixSize, blockSize);

        auto start = chrono::high_resolution_clock::now();
        solver.decompose(A);
        auto decompEnd = chrono::high_resolution_clock::now();
        Vector x = solver.solve(b);
        auto end = chrono::high_resolution_clock::now();

        double decompTime = chrono::duration<double>(decompEnd - start).count();
        double substTime = chrono::duration<double>(end - decompEnd).count();
        double totalTime = chrono::duration<double>(end - start).count();
        double speedup = serialTime / totalTime;
        double improvement = (1.0 - totalTime / serialTime) * 100.0;

        PerformanceMetrics metric;
        metric.decompositionTime = decompTime;
        metric.substitutionTime = substTime;
        metric.totalTime = totalTime;
        metric.speedup = speedup;
        metric.improvement = improvement;
        metric.version = "Hybrid CUDA";
        metric.blockSize = blockSize;
        metric.solutionError = verifySolution(A, x, b);
        metric.luError = verifyLUDecomposition(A, solver.getL(), solver.getU());
        results.push_back(metric);

        cout << "| Decomposition: " << fixed << setprecision(6) << decompTime << " s";
        cout << " (GPU)\n";
        cout << "| Substitution:  " << fixed << setprecision(6) << substTime << " s";
        cout << " (CPU)\n";
        cout << "| Total Time:    " << fixed << setprecision(6) << totalTime << " s\n";
        cout << "| Speedup:       " << fixed << setprecision(4) << speedup << "x\n";
        cout << "| Improvement:   " << fixed << setprecision(2) << improvement << "%\n";
        cout << "| Solution Error: " << scientific << setprecision(2) << metric.solutionError << endl;
        cout << "| LU Error:       " << scientific << setprecision(2) << metric.luError << endl;
        cout << "+----------------------------------------------------------------+\n\n";
    }

    // ========================================================================
    // PERFORMANCE SUMMARY TABLE
    // ========================================================================
    cout << "\n+==============================================================================================+\n";
    cout << "|                                    PERFORMANCE SUMMARY                                       |\n";
    cout << "+==============================================================================================+\n\n";

    cout << left
        << setw(16) << "Version"
        << setw(14) << "Block Size"
        << setw(16) << "Decomp (s)"
        << setw(16) << "Subst (s)"
        << setw(16) << "Total (s)"
        << setw(14) << "Speedup"
        << setw(14) << "Improvement\n";
    cout << string(106, '-') << "\n";

    for (const auto& m : results) {
        string blockInfo = (m.blockSize > 0) ? int_to_string(m.blockSize) : "N/A";
        cout << left
            << setw(16) << m.version
            << setw(14) << blockInfo
            << setw(16) << fixed << setprecision(6) << m.decompositionTime
            << setw(16) << fixed << setprecision(6) << m.substitutionTime
            << setw(16) << fixed << setprecision(6) << m.totalTime
            << setw(14) << fixed << setprecision(4) << m.speedup << "x"
            << setw(14) << fixed << setprecision(2) << m.improvement << "%\n";
    }
    cout << string(106, '-') << "\n\n";

    // ========================================================================
    // CORRECTNESS VERIFICATION
    // ========================================================================
    cout << "+================================================================+\n";
    cout << "|                   CORRECTNESS VERIFICATION                     |\n";
    cout << "+================================================================+\n\n";

    bool allCorrect = true;
    for (const auto& m : results) {
        string version = m.version;
        if (m.blockSize > 0) version += " (" + int_to_string(m.blockSize) + ")";

        bool solPass = m.solutionError < 1e-6;
        bool luPass = m.luError < 1e-6;

        cout << version << ":\n";
        cout << "  Solution Error: " << scientific << setprecision(2) << m.solutionError;
        cout << (solPass ? " [PASS]" : " [FAIL]") << "\n";
        cout << "  LU Error:       " << scientific << setprecision(2) << m.luError;
        cout << (luPass ? " [PASS]" : " [FAIL]") << "\n\n";

        if (!solPass || !luPass) allCorrect = false;
    }

    cout << (allCorrect ? "[OK] All implementations verified correct!\n\n" : "[ERROR] Some implementations failed!\n\n");

    // ========================================================================
    // PERFORMANCE INSIGHTS
    // ========================================================================
    cout << "+================================================================+\n";
    cout << "|                     PERFORMANCE INSIGHTS                       |\n";
    cout << "+================================================================+\n\n";

    // Find best configuration
    double bestSpeedup = 0.0;
    int bestIdx = 1;
    for (size_t i = 1; i < results.size(); i++) {
        if (results[i].speedup > bestSpeedup) {
            bestSpeedup = results[i].speedup;
            bestIdx = i;
        }
    }

    cout << "Best Configuration:\n";
    cout << "  Block Size: " << results[bestIdx].blockSize << " threads\n";
    cout << "  Speedup: " << fixed << setprecision(4) << bestSpeedup << "x\n";
    cout << "  Time Saved: " << fixed << setprecision(2) << results[bestIdx].improvement << "%\n\n";

    cout << "Architecture Notes:\n";
    cout << "  - Hybrid design: GPU for decomposition, CPU for substitution\n";
    cout << "  - Substitution kept on CPU (inherently sequential)\n";
    cout << "  - Optimal for matrices 500x500 and larger\n";
    cout << "  - Performance scales with matrix size\n\n";

    if (matrixSize < 1000) {
        cout << "Recommendation:\n";
        cout << "  ! Matrix size (" << matrixSize << "x" << matrixSize << ") is relatively small\n";
        cout << "  ! GPU overhead may dominate for small matrices\n";
        cout << "  - Try larger matrices (1000+ for better GPU utilization)\n\n";
    }

    cout << "===================================================================\n";
    cout << "                        ANALYSIS COMPLETE                          \n";
    cout << "===================================================================\n\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    cout << "\nInitializing CUDA...\n";

    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
        cerr << "Error: No CUDA devices found!\n";
        return 1;
    }

    cout << "CUDA initialized successfully!\n";

    // Test with multiple sizes
    vector<int> testSizes = { 10, 500 };

    for (int size : testSizes) {
        runComprehensiveAnalysis(size);
    }

    return 0;
}