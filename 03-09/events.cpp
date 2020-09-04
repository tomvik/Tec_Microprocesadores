#include <Windows.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define kNumThreads 16

double acum = 0;

clock_t start, end;
HANDLE event_handler;

DWORD WINAPI piFuncMutex(LPVOID pArg) {
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

int main() {
    start = clock();

    int y[kNumThreads];
    HANDLE hThread[kNumThreads];

    event_handler = CreateEvent(NULL, false, false, NULL);
    SetEvent(event_handler);

    for (int i = 0; i < kNumThreads; i++) {
        y[i] = i;

        hThread[i] = CreateThread(NULL, 0, piFuncMutex, &(y[i]), 0, NULL);
    }

    WaitForMultipleObjects(kNumThreads, hThread, TRUE, INFINITE);

    end = clock();

    printf("Resultado = %20.18lf \tcon eventos y con %d threads tomo: (%ld)ms\n", acum, kNumThreads,
           end - start);
    return 0;
}
