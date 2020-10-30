#include <omp.h>
#include <stdio.h>

int main() {
    int array[10];
    array[0] = 1;
    array[1] = 2;
    for (int i = 2; i < 10; i++) {
        array[i] = array[i - 1] + array[i - 2];
    }
    int i;
    int accum = 0;
#pragma omp parallel for
    for (i = 0; i < 10; i++) {
        accum += array[i];
    }

    int nthreads, procs, maxt, inpar, dynamic, nested;

#pragma omp parallel
    {
#pragma omp master
        {
            printf("Thread %d getting environment info...\n", omp_get_thread_num());

            procs = omp_get_num_procs();
            nthreads = omp_get_num_threads();
            maxt = omp_get_max_threads();
            inpar = omp_in_parallel();
            dynamic = omp_get_dynamic();
            nested = omp_get_nested();

            printf("Number of processors = %d\n", procs);
            printf("Number of threads = %d\n", nthreads);
            printf("Max threads = %d\n", maxt);
            printf("In parallel? = %d\n", inpar);
            printf("Dynamic threads enabled? = %d\n", dynamic);
            printf("Nested parallelism enabled? = %d\n", nested);
        }
    }

    printf("Last element value is % d\n", accum);
}