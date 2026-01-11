# Distributed & Parallel LU Decomposition Solver

## Project Overview

This project implements a **Linear Equation Solver** for systems in the form of  using **Doolittle’s LU Decomposition**. The solver is developed across four distinct computing frameworks to evaluate performance, scalability, and efficiency:

1. **Serial (CPU):** Baseline implementation.
2. **OpenMP (Shared Memory):** Multi-threaded CPU parallelization.
3. **MPI (Distributed Memory):** Multi-process parallelization for clusters.
4. **CUDA (Heterogeneous):** GPU-accelerated massive parallelism.

The goal is to compare how different parallel paradigms handle the cubic complexity () of matrix factorization.

> **Analogy:** Think of this project like trying to clean a massive mansion (the matrix).
> * **Serial:** You clean the whole house by yourself, room by room. It works, but it takes forever.
> * **OpenMP:** You invite 8 friends over. You all are in the same house (shared memory) and can easily shout to each other to coordinate who cleans which room.
> * **MPI:** You split the mansion into separate wings. Each cleaner is in a different wing and can't see the others; they have to use walkie-talkies (network messages) to coordinate. If the wings are small, you spend more time talking on the radio than cleaning!
> * **CUDA:** You unleash 1,000 tiny robot vacuums. Individually they are weak, but together they clean the entire floor plan in seconds.
> 
> 

---

**Course:** AMCS2103 Distributed Systems and Parallel Computing **Institution:** Tunku Abdul Rahman University of Management and Technology 

---

## Project Structure

```text
/
├── code/
│   ├── serial.cpp       # Baseline sequential C++ implementation
│   ├── openmp.cpp       # Shared-memory parallelization using OpenMP
│   ├── mpi.cpp          # Distributed-memory parallelization using MPI
│   └── CUDA.cu          # GPU-accelerated implementation using CUDA
├── Report.pdf           # Full project documentation and analysis
└── README.md            # Project documentation

```

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
