#!/bin/bash

mkdir -p result_mpi
mkdir -p result_seq
mkdir -p result_openmp

echo "Compiling sequential version..."
g++ -std=c++17 -O3 -I /usr/include/eigen3 kernel_svm_seq.cpp -o kernel_svm_seq
echo "Compiling OpenMP version..."
g++ -std=c++17 -O3 -fopenmp -I /usr/include/eigen3 kernel_svm_openmp.cpp -o kernel_svm_openmp
echo "Compiling MPI version..."
mpic++ kernel_svm_mpi.cpp -o kernel_svm_mpi -I /usr/include/eigen3 -std=c++17 -O3 -Wall

rows_values=(100 1000 2000)
thread_values=(2 8 12)
process_values=(2 4 6)

echo "Running sequential version..."
for r in "${rows_values[@]}"; do
    echo "Starting sequential SVM with $r rows..."
    ./kernel_svm_seq -r "$r" -s "$r" # Use test size = train size for simplicity
    echo "Completed sequential SVM with $r rows"
done

# echo "Running OpenMP version..."
# for r in "${rows_values[@]}"; do
#     for t in "${thread_values[@]}"; do
#         echo "Starting OpenMP SVM with $r rows and $t threads..."
#         ./kernel_svm_openmp -r "$r" -s "$r" -t "$t" # Use test size = train size
#         echo "Completed OpenMP SVM with $r rows and $t threads"
#     done
# done

echo "Running MPI version..."
for r in "${rows_values[@]}"; do
    for p in "${process_values[@]}"; do
        if [ "$p" -gt "$r" ] && [ "$r" -gt 0 ]; then
            echo "Skipping MPI SVM with $r rows and $p processes (p > r)"
            continue
        fi
        echo "Starting MPI SVM with $r rows and $p processes..."
        mpirun -np "$p" -quiet ./kernel_svm_mpi -r "$r" -s "$r"
        echo "Completed MPI SVM with $r rows and $p processes"
    done
done

echo "All tests completed"