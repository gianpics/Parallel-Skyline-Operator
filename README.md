# Parallel Skyline Operator - OpenMP and CUDA

Project for the *High Performance Computing* course (a.y. 2024/2025) at the University of Bologna.

This repository contains two parallel implementations of the skyline operator: a shared-memory CPU version using **OpenMP** and a GPU version using **CUDA**, along with a report analysing their scalability and efficiency.

## Background

Given a set of `N` points in a `D`-dimensional space, point `P_i` **dominates** `P_j` if every coordinate of `P_i` is greater than or equal to the corresponding coordinate of `P_j`, and at least one is strictly greater. The **skyline** is the set of points that are not dominated by any other point. The skyline operator is used in multi-criteria optimisation problems and is available as an extension of SQL.

The project uses the simple elimination algorithm: each point still in the set is compared against all others, and every point it dominates is discarded. What remains is the skyline. The algorithm is not the most efficient one, and its running time depends on `N`, `D` and the skyline size `K` (it is *output-sensitive*). The goal is to parallelise it efficiently, not to improve on the state of the art.

The task was to write two parallel implementations: one with **OpenMP** and one with **MPI or CUDA** (CUDA was chosen). The programs read from standard input and write to standard output:

```
Input:                          Output:
D [ignored text]                D
N                               K
P_00 P_01 ... P_0,D-1           (K skyline points, input order)
...
P_N-1,0 ... P_N-1,D-1
```

The complete assignment (in Italian) is in [ProgettoHPC2425-1.pdf](ProgettoHPC2425-1.pdf).

## Repository contents

| File | Description |
|---|---|
| `src/skyline.c` | Serial reference implementation (provided by the course) |
| `src/omp-skyline.c` | OpenMP version |
| `src/cuda-skyline.cu` | CUDA version |
| `src/hpc.h` | Helper header (provided by the course) |
| `src/Makefile` | Build file |
| `Relazione.pdf` | Report on the solutions and performance analysis (in Italian) |

## Build and run

Both versions can be built with the provided Makefile. The commands used are:

```sh
# OpenMP
gcc -std=c99 -Wall -Wpedantic -fopenmp -O2 omp-skyline.c -o omp-skyline

# CUDA
nvcc -Wno-deprecated-gpu-targets cuda-skyline.cu -o cuda-skyline -lm
```

Run:

```sh
./omp-skyline < input_file > output_file
./cuda-skyline < input_file > output_file
```

The number of OpenMP threads can be set with the `OMP_NUM_THREADS` environment variable.

## Parallelisation strategies

### OpenMP

The points are processed with a *partition* pattern: each thread is given a share of the points, and each point can be compared with the others independently of the rest of the data.

- The outer loop of the serial algorithm is parallelised with `omp parallel for`. The only shared state is the array marking each point as dominated or not. Reads of `s[j]` in the inner loop are protected with `omp atomic read`, to avoid conflicts with a concurrent write when a point is discarded.
- The cost of checking a point varies and cannot be predicted, so the outer loop uses **dynamic scheduling** with the default chunk size of 1.
- The skyline size is computed in a separate loop with a `reduction` and **static scheduling**, since every iteration costs the same.

### CUDA

The kernel follows the same approach as the OpenMP version: each thread computes its own index and checks whether the corresponding point belongs to the skyline. The solution array is initialised by a separate kernel, while the skyline points are counted serially.

## Performance evaluation

**Setup.** Programs were run on a server with an Intel Core i7-5820K at 3.30 GHz (6 physical cores with hyperthreading, 64 GB RAM) and two Nvidia GeForce Titan X GPUs (12 GB each). Each configuration was run three times and the mean time is reported. Every result was validated against ground-truth output produced by the serial program. The tests were automated with Bash scripts and the plots were generated with Python (matplotlib).

**OpenMP.** The implementation scales well with both the workload and the number of threads. Efficiency stays close to ideal while only physical cores are used (up to 6); beyond that, hyperthreading brings an overhead that slows the improvement. `test1` (3 dimensions, small skyline) benefits least from parallelism on every metric. `test7` (200 dimensions, largest skyline) shows a good speedup but degrades more sharply than the intermediate datasets at the maximum thread count. 

**CUDA.** Execution times are nearly identical for 128, 256, 512 and 1024 threads per block. This suggests the work is evenly spread across the streaming multiprocessors and occupancy is high. The remaining bottlenecks are the memory transfers and the serial sections. Since part of the code runs serially, the comparison is made against the serial CPU version. The GPU is much faster on the larger datasets. On `test1` it is slower than the serial version, because the problem is too small to offset the overhead.

**Conclusions.** OpenMP scales with the available cores, but its performance is gradually limited by overhead, especially from virtual cores. CUDA exploits the massive parallelism of the GPU and handles the datasets easily, though small instances do not benefit from it.

See [Relazione.pdf](Relazione.pdf) for the full analysis, with execution time, speedup and strong scaling efficiency plots.

## Notes

- Datasets from the course are not included. They were generated with a script included in the original assignment.
