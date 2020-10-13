#include <omp.h>
#include <stdio.h>

int main() {
    int rt = 9000;
    int at = 9000;

    do {
        rt++;
        omp_set_num_threads(rt);

#pragma omp parallel
        {
            if (omp_get_thread_num() == 0) {
                at = omp_get_num_threads();
                printf("Master program gave me %d threads! Nice!\n", at);
            }
        }
    } while (rt == at);

    printf("Master program gave me %d threads! Oh no!\n", at);
    return 0;
}