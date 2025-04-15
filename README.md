# Kernel SVM Parallel Implementations

This document provides instructions on how to compile and run the sequential, OpenMP, and MPI implementations of the Kernel SVM algorithm provided in this project.

## System Requirements

*   **OS:** Linux (tested on Ubuntu/Debian-based distributions)
*   **Compiler:** `g++` with C++17 support
*   **Libraries:**
    *   OpenMP (usually included with modern `g++`)
    *   MPI (e.g., OpenMPI)
    *   Eigen3

## Installation (Ubuntu/Debian)

1.  Update package list:
    ```bash
    sudo apt update
    ```

2.  Install build tools (including `g++` and OpenMP):
    ```bash
    sudo apt install build-essential -y
    ```

3.  Install OpenMPI:
    ```bash
    sudo apt install openmpi-bin libopenmpi-dev -y
    ```

4.  Install Eigen3 library:
    ```bash
    sudo apt install libeigen3-dev -y
    ```

## Compilation

The compilation process is handled automatically by the `run.sh` script. It compiles three versions:

1.  **Sequential:**
    ```bash
    g++ -std=c++17 -O3 -I /usr/include/eigen3 kernel_svm_seq.cpp -o kernel_svm_seq
    ```
2.  **OpenMP:**
    ```bash
    g++ -std=c++17 -O3 -fopenmp -I /usr/include/eigen3 kernel_svm_openmp.cpp -o kernel_svm_openmp
    ```
3.  **MPI:**
    ```bash
    mpic++ kernel_svm_mpi.cpp -o kernel_svm_mpi -I /usr/include/eigen3 -std=c++17 -O3 -Wall
    ```

## Execution

To compile all versions and run the benchmarks:

1.  Make the script executable (if needed):
    ```bash
    chmod +x run.sh
    ```

2.  Execute the script:
    ```bash
    ./run.sh
    ```

The script will:
*   Create output directories: `result_seq/`, `result_openmp/`, `result_mpi/`.
*   Compile the source files.
*   Run the sequential version for different data sizes.
*   Run the MPI version for different data sizes and process counts.
    > [!NOTE]
    > The OpenMP execution section is currently commented out in `run.sh`. You can uncomment it to run those tests as well.

## Output

Timing results and any other output from the programs will be saved in files within the respective `result_seq/`, `result_openmp/`, and `result_mpi/` directories. The filenames typically indicate the parameters used for the run (e.g., number of rows, threads, processes). 