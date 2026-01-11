# Distributed & Parallel LU Decomposition Solver

## Project Overview

This project implements a **Linear Equation Solver** for systems in the form of  using **Doolittle’s LU Decomposition**. The solver is developed across four distinct computing frameworks to evaluate performance, scalability, and efficiency:

1. **Serial (CPU):** Baseline implementation.
2. **OpenMP (Shared Memory):** Multi-threaded CPU parallelization.
3. **MPI (Distributed Memory):** Multi-process parallelization for clusters.
4. **CUDA (Heterogeneous):** GPU-accelerated massive parallelism.

---

## Prerequisites

To compile and run these solvers, you need the following installed on your system (tested on Windows 11):

1. **C++ Compiler:** GCC (g++) or MSVC.
2. **OpenMP:** Usually bundled with GCC; enabled via flags.
3. **MPI Library:** MS-MPI (Windows) or MPICH/OpenMPI (Linux).
4. 
**NVIDIA CUDA Toolkit:** `nvcc` compiler and compatible NVIDIA GPU (tested on RTX 4060).

## Methodology

The solvers use **Doolittle's Algorithm**, decomposing Matrix  into:

*  (Lower Triangular Matrix)
*  (Upper Triangular Matrix)

Such that . The system  is then solved via:

1. **Forward Substitution:** 
2. **Backward Substitution:** 

**Verification:** All implementations verify correctness by calculating the residual error  and reconstruction error .

---

## Performance Results

Based on a  matrix size:

| Implementation | Configuration | Execution Time (s) | Speedup | Note |
| --- | --- | --- | --- | --- |
| **CUDA** | 1024 threads/block | **0.029372** | **1.27x** | Best Performer |
| **OpenMP** | 8 threads | 0.050349 | 1.16x | Good efficiency |
| **Serial** | 1 thread | 0.058192 | 1.00x | Baseline |
| **MPI** | 8 processes | 0.188315 | 0.14x | High overhead for small N |

**Conclusion:**
For dense matrices of this size, **CUDA** offers the best performance due to massive parallelism. **MPI** suffers from communication bottlenecks when the problem size () is too small relative to the communication overhead.
