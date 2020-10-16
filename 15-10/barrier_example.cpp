#include <Windows.h>
#include <omp.h>
#include <stdio.h>

int main() {
    int x = 1;

#pragma omp parallel num_threads(4)
    {
        if (omp_get_thread_num() == 0) {
            Sleep(2);
            x = 13;
        }

#pragma omp barrier
        // Race condition when reading X.
        printf("Thread (%d): x = %d\n", omp_get_thread_num(), x);

#pragma omp barrier
        if (omp_get_thread_num() == 0) {  // No race condition.
            printf("Threads (%d): x = %d\n", omp_get_thread_num(), x);
        } else {
            printf("Threads (%d): x = %d\n", omp_get_thread_num(), x);
        }
    }
    return 0;
}
