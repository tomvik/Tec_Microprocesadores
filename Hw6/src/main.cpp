// Code made by Tomas Alejandro Lugo Salinas
// for the Hw6 of the lecture of Multiprocessors.
// Compiled with Cmake as follows:
//   cd Hw6
//   mkdir build && cd build
//   cmake .. -G"MinGW Makefiles"
//   cmake --build .
// Executed as: main.exe
// NOTE: If compilation fails, this link was extremely useful:
// https://stackoverflow.com/questions/35869564/cmake-on-windows

#include <Timer/Timer.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>

void sequencial(const float A[16], const float B[16], float C[16], float D[16], float E[16],
                float F[16]) {
    Timer::Timer timer("Sequencial");
    for (int i = 0; i < 16; ++i) {
        C[i] = A[i] + B[i];
        D[i] = A[i] - B[i];
        E[i] = A[i] * B[i];
        F[i] = A[i] / B[i];
    }
    printf("C= ");
    for (int i = 0; i < 16; ++i) {
        printf("%.2f ", C[i]);
    }
    printf("\n");
    printf("D= ");
    for (int i = 0; i < 16; ++i) {
        printf("%.2f ", D[i]);
    }
    printf("\n");
    printf("E= ");
    for (int i = 0; i < 16; ++i) {
        printf("%.2f ", E[i]);
    }
    printf("\n");
    printf("F= ");
    for (int i = 0; i < 16; ++i) {
        printf("%.2f ", F[i]);
    }
    printf("\n");
}

void OMP_Sections(const float A[16], const float B[16], float C[16], float D[16], float E[16],
                  float F[16]) {
    Timer::Timer timer("OMP_Sections");
    int i = 0;
#pragma omp parallel num_threads(4) shared(A, B, C, D, E, F) private(i)
    {
#pragma omp sections nowait
        {
#pragma omp section
            {
                printf("C thread: %d\n", omp_get_thread_num());
                for (i = 0; i < 16; ++i) {
                    C[i] = A[i] + B[i];
                }
            }
#pragma omp section
            {
                printf("D thread: %d\n", omp_get_thread_num());
                for (i = 0; i < 16; ++i) {
                    D[i] = A[i] - B[i];
                }
            }
#pragma omp section
            {
                printf("E thread: %d\n", omp_get_thread_num());
                for (i = 0; i < 16; ++i) {
                    E[i] = A[i] * B[i];
                }
            }
#pragma omp section
            {
                printf("F thread: %d\n", omp_get_thread_num());
                for (i = 0; i < 16; ++i) {
                    F[i] = A[i] / B[i];
                }
            }
        }
    }
    printf("C= ");
    for (i = 0; i < 16; ++i) {
        printf("%.2f ", C[i]);
    }
    printf("\n");
    printf("D= ");
    for (i = 0; i < 16; ++i) {
        printf("%.2f ", D[i]);
    }
    printf("\n");
    printf("E= ");
    for (i = 0; i < 16; ++i) {
        printf("%.2f ", E[i]);
    }
    printf("\n");
    printf("F= ");
    for (i = 0; i < 16; ++i) {
        printf("%.2f ", F[i]);
    }
    printf("\n");
}

int main() {
    const float A[16] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    const float B[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float C[16] = {0};
    float D[16] = {0};
    float E[16] = {0};
    float F[16] = {0};

    sequencial(A, B, C, D, E, F);

    for (int i = 0; i < 16; ++i) {
        C[i] = 0;
        D[i] = 0;
        E[i] = 0;
        F[i] = 0;
    }

    OMP_Sections(A, B, C, D, E, F);

    return 0;
}