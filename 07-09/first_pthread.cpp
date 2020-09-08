#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void *thread_routine(void *arg) {
    int *param = (int *)arg;
    printf("Hello Thread %d\n", *param);
}

void executeSingleThread() {
    int param = 0;
    pthread_t pthreadID;
    pthread_create(&pthreadID, NULL, thread_routine, &param);
    pthread_join(pthreadID, NULL);
}

void executeMultipleThreads(const int kNumThreads) {
    int param[kNumThreads];
    pthread_t pthreadID[kNumThreads];
    for (int i = 0; i < kNumThreads; ++i) {
        param[i] = i;
        pthread_create(&(pthreadID[i]), NULL, thread_routine, &(param[i]));
    }
    for (int i = 0; i < kNumThreads; ++i) {
        pthread_join(pthreadID[i], NULL);
        // pthread_detach(pthreadID[i]);
    }
}

int main() {
    executeMultipleThreads(4);
    exit(0);
}