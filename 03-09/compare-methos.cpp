#include <Windows.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define kNumThreads 16
#define kIterations 100

double acum = 0;

clock_t start, end;

struct piParams {
    int id = 0;
    double acum = 0;
};

CRITICAL_SECTION CriticalSection;

HANDLE event_handler;

DWORD WINAPI piFuncParams(LPVOID pArg) {
    piParams *params = reinterpret_cast<piParams *>(pArg);

    double x;
    int64_t i;
    double baseIntervalo;
    double fdx;
    int64_t cantidadIntervalos = 1000000000;
    baseIntervalo = 1.0 / cantidadIntervalos;
    double local_acum = 0;

    x = baseIntervalo * (cantidadIntervalos / kNumThreads) * params->id;

    for (i = 0; i < cantidadIntervalos / kNumThreads; i++) {
        fdx = 4.0 / (1 + x * x);
        local_acum += (fdx * baseIntervalo);
        x = x + baseIntervalo;
    }

    params->acum = local_acum;

    return 0;
}

DWORD WINAPI piFuncCriticalSection(LPVOID pArg) {
    int *id = reinterpret_cast<int *>(pArg);

    double x;
    int64_t i;
    double baseIntervalo;
    double fdx;
    int64_t cantidadIntervalos = 1000000000;
    double localAcum = 0;
    baseIntervalo = 1.0 / cantidadIntervalos;

    x = baseIntervalo * (cantidadIntervalos / kNumThreads) * *id;

    for (i = 0; i < cantidadIntervalos / kNumThreads; i++) {
        fdx = 4.0 / (1 + x * x);
        localAcum += (fdx * baseIntervalo);
        x = x + baseIntervalo;
    }

    EnterCriticalSection(&CriticalSection);
    acum += localAcum;
    LeaveCriticalSection(&CriticalSection);

    return 0;
}

DWORD WINAPI piFuncEvents(LPVOID pArg) {
    int *id = reinterpret_cast<int *>(pArg);

    double x;
    int64_t i;
    double baseIntervalo;
    double fdx;
    int64_t cantidadIntervalos = 1000000000;
    double localAcum = 0;
    baseIntervalo = 1.0 / cantidadIntervalos;

    x = baseIntervalo * (cantidadIntervalos / kNumThreads) * *id;

    for (i = 0; i < cantidadIntervalos / kNumThreads; i++) {
        fdx = 4.0 / (1 + x * x);
        localAcum += (fdx * baseIntervalo);
        x = x + baseIntervalo;
    }

    WaitForSingleObject(event_handler, INFINITE);
    acum += localAcum;
    SetEvent(event_handler);

    return 0;
}

int64_t runEvents() {
    acum = 0;

    start = clock();

    int y[kNumThreads];
    HANDLE hThread[kNumThreads];

    event_handler = CreateEvent(NULL, false, false, NULL);
    SetEvent(event_handler);

    for (int i = 0; i < kNumThreads; i++) {
        y[i] = i;

        hThread[i] = CreateThread(NULL, 0, piFuncEvents, &(y[i]), 0, NULL);
    }

    WaitForMultipleObjects(kNumThreads, hThread, TRUE, INFINITE);

    end = clock();

    return end - start;
}

int64_t runCriticalSection() {
    acum = 0;

    start = clock();

    int y[kNumThreads];
    HANDLE hThread[kNumThreads];

    InitializeCriticalSection(&CriticalSection);

    for (int i = 0; i < kNumThreads; i++) {
        y[i] = i;

        hThread[i] = CreateThread(NULL, 0, piFuncCriticalSection, &(y[i]), 0, NULL);
    }

    WaitForMultipleObjects(kNumThreads, hThread, TRUE, INFINITE);

    DeleteCriticalSection(&CriticalSection);

    end = clock();

    return end - start;
}

int64_t runParameters() {
    acum = 0;

    start = clock();

    piParams y[kNumThreads];
    HANDLE hThread[kNumThreads];

    for (int i = 0; i < kNumThreads; i++) {
        y[i].id = i;

        hThread[i] = CreateThread(NULL, 0, piFuncParams, &y[i], 0, NULL);
    }

    WaitForMultipleObjects(kNumThreads, hThread, TRUE, INFINITE);

    for (int i = 0; i < kNumThreads; i++) {
        acum += y[i].acum;
    }

    end = clock();

    return end - start;
}

int main() {
    printf("iteraciones por metodo: %d y usando %d threads\n", kIterations, kNumThreads);

    int64_t time = 0;

    for (int i = 0; i < kIterations; ++i) {
        time += runParameters();
    }

    time /= kIterations;

    printf("Tiempo con parametros:\t\t tomo: (%ld)ms\n", time);

    time = 0;
    for (int i = 0; i < kIterations; ++i) {
        time += runCriticalSection();
    }

    time /= kIterations;

    printf("Tiempo con critical section:\t tomo: (%ld)ms\n", time);

    time = 0;
    for (int i = 0; i < kIterations; ++i) {
        time += runEvents();
    }

    time /= kIterations;

    printf("Tiempo con eventos:\t\t tomo: (%ld)ms\n", time);
    return 0;
}
