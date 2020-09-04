#include <Windows.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define kNumThreads 16
#define kWithParams false

double acum = 0;

clock_t start, end;
struct piParams {
    int id = 0;
    double acum = 0;
};

DWORD WINAPI piFunc(LPVOID pArg) {
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

int main() {
    start = clock();

    piParams y[kNumThreads];
    HANDLE hThread[kNumThreads];


    for (int i = 0; i < kNumThreads; i++) {
        y[i].id = i;

        hThread[i] = CreateThread(NULL, 0, piFunc, &y[i], 0, NULL);
    }

    WaitForMultipleObjects(kNumThreads, hThread, TRUE, INFINITE);

    for (int i = 0; i < kNumThreads; i++) {
        acum += y[i].acum;
    }

    end = clock();

    printf("Resultado = %20.18lf \tcon parametros y con %d threads tomo: (%ld)ms\n", acum,
           kNumThreads, end - start);
    return 0;
}
