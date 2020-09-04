#include <Windows.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define kNumThreads 4
#define kWithParams false

double acum = 0;

clock_t start, end;
struct piParams {
    int id = 0;
    double acum = 0;
};

#if kWithParams

DWORD WINAPI piFunc(LPVOID pArg) {
    piParams *params = reinterpret_cast<piParams *>(pArg);

    double x;
    int64_t i;
    double baseIntervalo;
    double fdx;
    int64_t cantidadIntervalos = 1000000000;
    baseIntervalo = 1.0 / cantidadIntervalos;

    x = baseIntervalo * (cantidadIntervalos / kNumThreads) * params->id;

    params->acum = 0;
    for (i = 0; i < cantidadIntervalos / kNumThreads; i++) {
        fdx = 4.0 / (1 + x * x);
        params->acum = params->acum + (fdx * baseIntervalo);
        x = x + baseIntervalo;
    }

    return 0;
}

#else
CRITICAL_SECTION CriticalSection;

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
    EnterCriticalSection(&CriticalSection);
    acum += localAcum;
    LeaveCriticalSection(&CriticalSection);

    return 0;
}
#endif

int main() {
    start = clock();

    piParams y[kNumThreads];
    HANDLE hThread[kNumThreads];

#if !kWithParams
    InitializeCriticalSection(&CriticalSection);
#endif

    for (int i = 0; i < kNumThreads; i++) {
        y[i].id = i;

#if kWithParams
        hThread[i] = CreateThread(NULL, 0, piFunc, &y[i], 0, NULL);
#else
        hThread[i] = CreateThread(NULL, 0, piFuncMutex, &(y[i].id), 0, NULL);
#endif
    }

    WaitForMultipleObjects(kNumThreads, hThread, TRUE, INFINITE);

#if kWithParams
    for (int i = 0; i < kNumThreads; i++) {
        acum += y[i].acum;
    }
#endif

    end = clock();

#if !kWithParams
    DeleteCriticalSection(&CriticalSection);
#endif

#if kWithParams
    printf("Resultado = %20.18lf \tcon parametros y con %d threads tomo: (%ld)ms\n", acum,
           kNumThreads, end - start);
#else
    printf("Resultado = %20.18lf \tcon critical section y con %d threads tomo: (%ld)ms\n", acum,
           kNumThreads, end - start);
#endif
    return 0;
}
