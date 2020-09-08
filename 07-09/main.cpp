#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

constexpr int kNumThreads = 4;

struct piParams {
    int id = 0;
    double acum = 0;
};

void *singlePi(void *arg) {
    piParams *params = reinterpret_cast<piParams *>(arg);

    const int64_t cantidadIntervalos = 1000000000;
    const double baseIntervalo = 1.0 / cantidadIntervalos;

    double x = baseIntervalo * (cantidadIntervalos / kNumThreads) * params->id;
    double fdx;
    double local_acum = 0;

    for (int64_t i = 0; i < cantidadIntervalos / kNumThreads; i++) {
        fdx = 4.0 / (1 + x * x);
        local_acum += (fdx * baseIntervalo);
        x = x + baseIntervalo;
    }

    // printf("local acum: %lf\n", local_acum);

    params->acum = local_acum;

    return NULL;
}

double calcPiParams() {
    piParams params[kNumThreads];
    pthread_t pthreadID[kNumThreads];
    for (int i = 0; i < kNumThreads; ++i) {
        params[i].id = i;
        pthread_create(&(pthreadID[i]), NULL, singlePi, &(params[i]));
    }

    double acum = 0;
    for (int i = 0; i < kNumThreads; ++i) {
        pthread_join(pthreadID[i], NULL);
        acum += params[i].acum;
    }

    return acum;
}

void calcPi() {
    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_REALTIME, &start);

    const double acum = calcPiParams();

    clock_gettime(CLOCK_REALTIME, &finish);
    elapsed = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1000000000L;
    printf("Resultado = %20.18lf con %d threads y (%lf) segundos\n", acum, kNumThreads, elapsed);
}

int main() {
    calcPi();
    return 0;
}
