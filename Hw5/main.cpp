// Code made by Tomas Alejandro Lugo Salinas
// for the Hw5 of the lecture of Multiprocessors.
// Compiled with: g++ -std=c++14 -fopenmp -g main.cpp -o main.exe
// Executed as: ./main.exe

#include <omp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

constexpr int kNumThreads = 16;
constexpr int kNumRunTypes = 4;
constexpr int kNumRunAmounts = 5;

const char *const kRunTypes[] = {"SingleThread", "PThread", "OMPThread", "AllModes"};

enum class RunType { SingleThread = 0, PThread = 1, OMPThread = 2, AllModes = 3 };

struct piParams {
    int id = 0;
    double acum = 0;
};

double calcPiSingle() {
    const int64_t cantidadIntervalos = 1000000000;
    const double baseIntervalo = 1.0 / cantidadIntervalos;

    double x = 0;
    double fdx;
    double local_acum = 0;

    for (int64_t i = 0; i < cantidadIntervalos; i++) {
        fdx = 4.0 / (1 + x * x);
        local_acum += (fdx * baseIntervalo);
        x = x + baseIntervalo;
    }

    return local_acum;
}

void *calcPiSinglePThread(void *arg) {
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

    params->acum = local_acum;

    return NULL;
}

double calcPiPThread() {
    piParams params[kNumThreads];
    pthread_t pthreadID[kNumThreads];
    for (int i = 0; i < kNumThreads; ++i) {
        params[i].id = i;
        pthread_create(&(pthreadID[i]), NULL, calcPiSinglePThread, &(params[i]));
    }

    double acum = 0;
    for (int i = 0; i < kNumThreads; ++i) {
        pthread_join(pthreadID[i], NULL);
        acum += params[i].acum;
    }

    return acum;
}

double calcOMPPThread() {
    double total_acum = 0;

#pragma omp parallel num_threads(kNumThreads)
    {
        const int64_t cantidadIntervalos = 1000000000;
        const double baseIntervalo = 1.0 / cantidadIntervalos;
        const int numThreads = omp_get_num_threads();
        double x = baseIntervalo * (cantidadIntervalos / numThreads) * omp_get_thread_num();

        double fdx;
        double local_acum = 0;

        const int64_t limIntervalo = cantidadIntervalos / numThreads;
        for (int64_t i = 0; i < limIntervalo; i++) {
            fdx = 4.0 / (1 + x * x);
            local_acum += (fdx * baseIntervalo);
            x = x + baseIntervalo;
        }

        total_acum += local_acum;
    }
    return total_acum;
}

void calcPi(const RunType run_type, const int run_amounts) {
    struct timespec start, finish;
    double elapsed, acum_elapsed = 0, pi_value;

    printf("Running for %s with %d Threads and average of %d:\n",
           kRunTypes[static_cast<int>(run_type)], kNumThreads, run_amounts);

    for (int i = 0; i < run_amounts; ++i) {
        clock_gettime(CLOCK_REALTIME, &start);

        switch (run_type) {
            case RunType::SingleThread:
                pi_value = calcPiSingle();
                break;
            case RunType::PThread:
                pi_value = calcPiPThread();
                break;
            case RunType::OMPThread:
                pi_value = calcOMPPThread();
                break;
            default:
                printf("[ERROR] type does not exist\n");
                break;
        }

        clock_gettime(CLOCK_REALTIME, &finish);
        elapsed = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
        acum_elapsed += elapsed;

        printf("**For run %d the result was %lf and time was: %lf seconds\n", i, pi_value, elapsed);
    }

    printf("*** Average time of %d runs is: %lf seconds\n", run_amounts,
           acum_elapsed / run_amounts);
}

int main() {
    int input_run_type = 0;

    printf(
        "Hello! Welcome to the super duper complex quantum program that calculates pi. Enjoy "
        ":)\n\n");
    for (int i = 0; i < kNumRunTypes; ++i) {
        printf("\tInput %d to select the mode %s\n", i, kRunTypes[i]);
    }

    scanf("%d", &input_run_type);
    const RunType run_type = static_cast<RunType>(input_run_type);

    if (run_type == RunType::AllModes) {
        calcPi(RunType::SingleThread, kNumRunAmounts);
        calcPi(RunType::PThread, kNumRunAmounts);
        calcPi(RunType::OMPThread, kNumRunAmounts);
    } else {
        calcPi(run_type, kNumRunAmounts);
    }

    return 0;
}
