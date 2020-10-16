#include <stdio.h>
#include <time.h>

long cantidadIntervalos = 1000;
double baseIntervalo;
double fdx;
double acum = 0;
clock_t start, end;

void left_rectangle(long long cantidad_intervalos) {
    double x;
    long i;
    baseIntervalo = 1.0 / cantidad_intervalos;
    for (i = 0, x = 0.0; i < cantidad_intervalos; i++) {
        fdx = 4 / (1 + x * x);
        acum = acum + (fdx * baseIntervalo);
        x = x + baseIntervalo;
    }
}

void middle_rectangle(long long cantidad_intervalos) {
    double x;
    long i;
    baseIntervalo = 1.0 / cantidad_intervalos;
    for (i = 0, x = baseIntervalo / 2; i < cantidad_intervalos; i++) {
        fdx = 4 / (1 + x * x);
        acum = acum + (fdx * baseIntervalo);
        x = x + baseIntervalo;
    }
}

int main() {
    for(int i = 0; i < 1000000000; ++i) {
        //start = clock();
        middle_rectangle(cantidadIntervalos+i);
        //end = clock();
        //printf("Resultado = %20.18lf (%ld)\n", acum, end - start);
        if(acum <= 3.141592659 && acum >= 3.14159265) {
            printf("%d\n", cantidadIntervalos+i);
            acum = 0;
            break;
        }
        acum = 0;
    }
    for(long long i = 2001900000; i > 0; --i) {
        //start = clock();
        acum = 0;
        left_rectangle(cantidadIntervalos+i);
        //end = clock();
        //printf("Resultado = %20.18lf (%ld)\n", acum, end - start);
        if(acum < 3.14159266 && acum >= 3.141592650) {
            continue;
        } else {
            printf("Resultado = %20.18lf (%ld)\n", acum, 0);
            printf("%d\n", cantidadIntervalos+i);
            break;
        }
        acum = 0;
    }

    return 0;
}
