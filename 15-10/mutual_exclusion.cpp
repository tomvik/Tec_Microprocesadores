#include <omp.h>
#include <stdio.h>
#include <time.h>
#define MAXTHREADS 8
long cantidadIntervalos = 10000000;  // 10 Million
double baseIntervalo;
// double acum = 0; //No puede ser una variable global
clock_t start, end;

int main() {
    int THREADS = MAXTHREADS;
    baseIntervalo = 1.0 / (double)cantidadIntervalos;

    double totalSum = 0;
    omp_set_num_threads(THREADS);
    start = clock();
#pragma omp parallel
    {
        int numThread = omp_get_thread_num();
        double acum = 0;  // No puede ser una variable global. Es una variable privada al thread.
        double fdx = 0;   // No puede ser una variable global. Es una variable privada al thread.
        double x;
        for (long i = numThread; i < cantidadIntervalos; i += THREADS) {
            x = i * baseIntervalo;
            fdx = 4 / (1 + x * x);
            acum += fdx;
        }
        acum *= baseIntervalo;  // Multiplico todas las alturas de los rectangulos acumuladas por el
                                // tamaño de la base.
#pragma omp atomic
        totalSum += acum;
        // printf("Resultado parcial (Thread %d)\nacum = %lf\n", numThread, acum);
    }
    end = clock();

    printf("\nResultado (%d threads) = %20.18lf (%ld)\n", THREADS, totalSum, end - start);
    return 0;
}
