#include <iostream>
#include <omp.h>

int main() {
    const int N = 10;
    int arr[N];
    for (int i = 0; i < N; ++i) arr[i] = i + 1; // arr = [1, 2, ..., 10]
    
    int sum = 0;

    auto parallel_sum = [&](const int* arr, int N, int& sum) {
        // All threads must call this function!
        #pragma omp for reduction(+:sum)
        for (int i = 0; i < N; ++i) {
            sum += arr[i];
        }
    };

    #pragma omp parallel
    {
        // Setup: only one thread does this
        #pragma omp single
        {
            std::cout << "Single thread (id " << omp_get_thread_num() << ") does setup.\n";
        }

        // All threads participate in the sum
        parallel_sum(arr, N, sum);

        // Only one thread does this
        #pragma omp single
        {
            std::cout << "Single thread (id " << omp_get_thread_num() << ") does cleanup.\n";
        }
        #pragma omp barrier
        #pragma omp single
        {
            std::cout << "Sum is: " << sum << std::endl; // Should print 55
        }
    }

    return 0;
}
