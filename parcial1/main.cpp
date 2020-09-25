#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

constexpr int kNumThreads = 4;
constexpr int kLength = 400;
constexpr int kBatchSize = kLength / kNumThreads;

int *numbers;

struct threeParams {
    int id = 0;
    double acum = 0;
};

int serialThreeParams() {
    int count = 0;
    for (int i = 0; i < kLength; i++) {
        if (numbers[i] == 3) count++;
    }
    return count;
}

void *singleThreeParams(void *arg) {
    threeParams *params = reinterpret_cast<threeParams *>(arg);

    const int first_index = params->id * kBatchSize;
    const int last_index = params->id == kNumThreads ? kLength : first_index + kBatchSize;

    int local_acum = 0;

    for (int i = first_index; i < last_index; ++i) {
        if (numbers[i] == 3) {
            ++local_acum;
        }
    }

    params->acum = local_acum;

    return NULL;
}

int calcthreeParams() {
    threeParams params[kNumThreads];
    pthread_t pthreadID[kNumThreads];
    for (int i = 0; i < kNumThreads; ++i) {
        params[i].id = i;
        pthread_create(&(pthreadID[i]), NULL, singleThreeParams, &(params[i]));
    }

    int acum = 0;
    for (int i = 0; i < kNumThreads; ++i) {
        pthread_join(pthreadID[i], NULL);
        acum += params[i].acum;
    }

    return acum;
}

void countThrees() {
    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_REALTIME, &start);

    const int acum = calcthreeParams();

    clock_gettime(CLOCK_REALTIME, &finish);
    elapsed = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1000000000L;
    printf("Resultado = %d con %d threads y (%lf) segundos\n", acum, kNumThreads, elapsed);
}

int main() {
    size_t datasize = sizeof(int) * kLength;
    numbers = reinterpret_cast<int *>(malloc(datasize));

    for (int i = 0; i < kLength; ++i) {
        numbers[i] = 0;
    }
    numbers[0] = 3;
    numbers[300] = 3;
    numbers[20] = 3;
    numbers[15] = 3;
    numbers[399] = 3;
    numbers[200] = 3;

    countThrees();
    return 0;
}
