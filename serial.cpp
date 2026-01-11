#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <random>
using namespace std;
using namespace std::chrono;

const int MAX_SIZE = 1000;  // Increased for larger tests

// Generate a random matrix with values between minVal and maxVal
vector<vector<double>> generateRandomMatrix(int N, double minVal = -10.0, double maxVal = 10.0) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(minVal, maxVal);

    vector<vector<double>> A(N, vector<double>(N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = dist(gen);
        }
        // Make diagonal dominant to ensure non-singular matrix
        A[i][i] += (maxVal - minVal) * N;
    }
    return A;
}

// Generate random vector b
vector<double> generateRandomVector(int N, double minVal = -10.0, double maxVal = 10.0) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(minVal, maxVal);

    vector<double> b(N);
    for (int i = 0; i < N; i++) {
        b[i] = dist(gen);
    }
    return b;
}

void doolittleLU(const vector<vector<double>>& A, const vector<double>& b, bool verbose = false) {
    int N = A.size();
    vector<vector<double>> L(N, vector<double>(N, 0));
    vector<vector<double>> U(N, vector<double>(N, 0));
    vector<double> y(N, 0);
    vector<double> x(N, 0);

    if (verbose) {
        cout << "Starting Doolittle LU Decomposition...\n";
    }

    auto start_time = high_resolution_clock::now();

    // LU Decomposition
    for (int i = 0; i < N; i++) {
        // Compute U row
        for (int j = i; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < i; k++)
                sum += L[i][k] * U[k][j];
            U[i][j] = A[i][j] - sum;

            if (verbose && N <= 5) {
                cout << "U[" << i << "][" << j << "] = " << A[i][j] << " - " << sum << " = " << U[i][j] << endl;
            }
        }

        L[i][i] = 1;

        if (verbose && N <= 5) {
            cout << "L[" << i << "][" << i << "] = 1\n";
        }

        // Compute L column
        for (int j = i + 1; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < i; k++)
                sum += L[j][k] * U[k][i];

            if (U[i][i] == 0) {
                cerr << "Zero pivot encountered at position (" << i << "," << i << ")! Cannot proceed.\n";
                return;
            }

            L[j][i] = (A[j][i] - sum) / U[i][i];

            if (verbose && N <= 5) {
                cout << "L[" << j << "][" << i << "] = (" << A[j][i] << " - " << sum << ") / " << U[i][i] <<
                    " = " << L[j][i] << endl;
            }
        }
    }

    auto lu_end_time = high_resolution_clock::now();

    if (verbose) {
        // Display L and U only for small matrices
        if (N <= 10) {
            cout << "\nLower Triangular Matrix L:\n";
            for (const auto& row : L) {
                for (double val : row)
                    cout << setw(12) << fixed << setprecision(4) << val;
                cout << endl;
            }

            cout << "\nUpper Triangular Matrix U:\n";
            for (const auto& row : U) {
                for (double val : row)
                    cout << setw(12) << fixed << setprecision(4) << val;
                cout << endl;
            }
        }

        // Forward substitution to solve Ly = b
        if (N <= 5) {
            cout << "\nForward Substitution to solve Ly = b:\n";
        }
    }

    for (int i = 0; i < N; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++)
            sum += L[i][j] * y[j];
        y[i] = b[i] - sum;

        if (verbose && N <= 5) {
            cout << "y[" << i << "] = " << b[i] << " - " << sum << " = " << y[i] << endl;
        }
    }

    // Backward substitution to solve Ux = y
    if (verbose && N <= 5) {
        cout << "\nBackward Substitution to solve Ux = y:\n";
    }

    for (int i = N - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < N; j++)
            sum += U[i][j] * x[j];

        if (U[i][i] == 0) {
            cerr << "Zero pivot encountered during back substitution!\n";
            return;
        }

        x[i] = (y[i] - sum) / U[i][i];

        if (verbose && N <= 5) {
            cout << "x[" << i << "] = (" << y[i] << " - " << sum << ") / " << U[i][i] << " = " << x[i] << endl;
        }
    }

    auto end_time = high_resolution_clock::now();

    auto lu_duration = duration<double>(lu_end_time - start_time);
    auto total_duration = duration<double>(end_time - start_time);

    // Display performance results
    cout << "\n===============================================\n";
    cout << "Performance Results for N = " << N << ":\n";
    cout << "===============================================\n";
    cout << "LU Decomposition Time: " << fixed << setprecision(6) << lu_duration.count() << " seconds\n";
    cout << "Total Solution Time: " << fixed << setprecision(6) << total_duration.count() << " seconds\n";
    cout << "Forward/Backward Substitution Time: " << fixed << setprecision(6) << (total_duration - lu_duration).count() << " seconds\n";

    // Display final solution only for small matrices
    if (verbose && N <= 10) {
        cout << "\nFirst 10 elements of solution (x):\n";
        int displayCount = min(10, N);
        for (int i = 0; i < displayCount; i++) {
            cout << "x[" << i << "] = " << fixed << setprecision(6) << x[i] << endl;
        }
    }

    // Verify solution for small matrices
    if (N <= 100) {
        double max_error = 0;
        for (int i = 0; i < N; i++) {
            double sum = 0;
            for (int j = 0; j < N; j++) {
                sum += A[i][j] * x[j];
            }
            double error = abs(sum - b[i]);
            if (error > max_error) {
                max_error = error;
            }
        }
        cout << "Maximum error |Ax - b|: " << scientific << setprecision(2) << max_error << endl;
    }

    cout << "===============================================\n";
}

void benchmarkSizes() {
    vector<int> sizes = { 10, 50, 100, 200, 500, 1000 };

    cout << "\n===============================================\n";
    cout << "BENCHMARKING DOOLITTLE LU DECOMPOSITION\n";
    cout << "===============================================\n\n";

    for (int N : sizes) {
        if (N > MAX_SIZE) {
            cout << "Skipping N=" << N << " (exceeds MAX_SIZE=" << MAX_SIZE << ")\n";
            continue;
        }

        cout << "\nTesting matrix size: " << N << "x" << N << endl;
        cout << "Generating random matrix and vector...\n";

        auto A = generateRandomMatrix(N);
        auto b = generateRandomVector(N);

        bool verbose = (N <= 10);  // Show details only for small matrices

        doolittleLU(A, b, verbose);

        // Clear memory
        A.clear();
        b.clear();

        cout << endl;
    }
}

int main() {
    int choice;

    cout << "===============================================\n";
    cout << "Doolittle LU Decomposition Solver\n";
    cout << "===============================================\n";
    cout << "Options:\n";
    cout << "1. Test specific matrix size\n";
    cout << "2. Run benchmark (multiple sizes)\n";
    cout << "3. Enter matrix manually\n";
    cout << "Enter your choice: ";
    cin >> choice;

    switch (choice) {
    case 1: {
        int N;
        cout << "Enter matrix size N (max " << MAX_SIZE << "): ";
        cin >> N;

        if (N > MAX_SIZE || N <= 0) {
            cerr << "Invalid matrix size.\n";
            return 1;
        }

        bool verbose;
        cout << "Show detailed output? (0=no, 1=yes): ";
        cin >> verbose;

        auto A = generateRandomMatrix(N);
        auto b = generateRandomVector(N);

        cout << "\nGenerated " << N << "x" << N << " matrix and vector.\n";

        if (N <= 10 && verbose) {
            cout << "\nMatrix A (first 10x10 if larger):\n";
            int displaySize = min(10, N);
            for (int i = 0; i < displaySize; i++) {
                for (int j = 0; j < displaySize; j++) {
                    cout << setw(10) << fixed << setprecision(2) << A[i][j];
                }
                if (N > 10) cout << " ...";
                cout << endl;
            }

            cout << "\nVector b (first 10 elements):\n";
            for (int i = 0; i < min(10, N); i++) {
                cout << "b[" << i << "] = " << b[i] << endl;
            }
        }

        doolittleLU(A, b, verbose);
        break;
    }

    case 2:
        benchmarkSizes();
        break;

    case 3: {
        int N;
        cout << "Enter the size of the matrix (max " << MAX_SIZE << "): ";
        cin >> N;

        if (N > MAX_SIZE || N <= 0) {
            cerr << "Invalid matrix size.\n";
            return 1;
        }

        vector<vector<double>> A(N, vector<double>(N));
        vector<double> b(N);

        cout << "Enter the elements of matrix A (" << N << "x" << N << "):\n";
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                cin >> A[i][j];

        cout << "Enter the elements of vector b:\n";
        for (int i = 0; i < N; i++)
            cin >> b[i];

        bool verbose = true;
        doolittleLU(A, b, verbose);
        break;
    }

    default:
        cerr << "Invalid choice.\n";
        return 1;
    }

    return 0;
}