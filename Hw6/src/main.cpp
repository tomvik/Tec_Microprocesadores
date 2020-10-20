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

constexpr int kArraySize = 16;

void sequencial(const float A[kArraySize], const float B[kArraySize], float C[kArraySize],
                float D[kArraySize], float E[kArraySize], float F[kArraySize]) {
    Timer::Timer timer("Sequencial");
    for (int i = 0; i < kArraySize; ++i) {
        C[i] = A[i] + B[i];
        D[i] = A[i] - B[i];
        E[i] = A[i] * B[i];
        F[i] = A[i] / B[i];
    }
    printf("C= ");
    for (int i = 0; i < kArraySize; ++i) {
        printf("%.2f ", C[i]);
    }
    printf("\n");
    printf("D= ");
    for (int i = 0; i < kArraySize; ++i) {
        printf("%.2f ", D[i]);
    }
    printf("\n");
    printf("E= ");
    for (int i = 0; i < kArraySize; ++i) {
        printf("%.2f ", E[i]);
    }
    printf("\n");
    printf("F= ");
    for (int i = 0; i < kArraySize; ++i) {
        printf("%.2f ", F[i]);
    }
    printf("\n");
}

void OMP_Sections(const float A[kArraySize], const float B[kArraySize], float C[kArraySize],
                  float D[kArraySize], float E[kArraySize], float F[kArraySize]) {
    Timer::Timer timer("OMP_Sections");
    int i = 0;
#pragma omp parallel num_threads(4) shared(C, D, E, F) private(i)  // nowait
    {
#pragma omp sections nowait
        {
#pragma omp section
            {
                printf("C thread: %d\n", omp_get_thread_num());
                for (i = 0; i < kArraySize; ++i) {
                    C[i] = A[i] + B[i];
                }
            }  // End section.
#pragma omp section
            {
                printf("D thread: %d\n", omp_get_thread_num());
                for (i = 0; i < kArraySize; ++i) {
                    D[i] = A[i] - B[i];
                }
            }  // End section.
#pragma omp section
            {
                printf("E thread: %d\n", omp_get_thread_num());
                for (i = 0; i < kArraySize; ++i) {
                    E[i] = A[i] * B[i];
                }
            }  // End section.
#pragma omp section
            {
                printf("F thread: %d\n", omp_get_thread_num());
                for (i = 0; i < kArraySize; ++i) {
                    F[i] = A[i] / B[i];
                }
            }  // End section.
        }      // End sections

    }  // End Parallel

    printf("C= ");
    for (i = 0; i < kArraySize; ++i) {
        printf("%.2f ", C[i]);
    }
    printf("\n");
    printf("D= ");
    for (i = 0; i < kArraySize; ++i) {
        printf("%.2f ", D[i]);
    }
    printf("\n");
    printf("E= ");
    for (i = 0; i < kArraySize; ++i) {
        printf("%.2f ", E[i]);
    }
    printf("\n");
    printf("F= ");
    for (i = 0; i < kArraySize; ++i) {
        printf("%.2f ", F[i]);
    }
    printf("\n");
}

int main() {
    float A[kArraySize] = {0};
    float B[kArraySize] = {0};
    float C[kArraySize] = {0};
    float D[kArraySize] = {0};
    float E[kArraySize] = {0};
    float F[kArraySize] = {0};

    for (int i = 0; i < kArraySize; ++i) {
        A[i] = 10 + i;
        B[i] = 1 + i;
    }

    sequencial(A, B, C, D, E, F);

    for (int i = 0; i < kArraySize; ++i) {
        C[i] = 0;
        D[i] = 0;
        E[i] = 0;
        F[i] = 0;
    }

    OMP_Sections(A, B, C, D, E, F);

    return 0;
}