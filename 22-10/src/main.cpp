// Code made by Tomas Alejandro Lugo Salinas
// for the Hw6 of the lecture of Multiprocessors.
// Compiled with Cmake in Windows as follows:
//   cd Hw6
//   mkdir build && cd build
//   cmake .. -G"MinGW Makefiles"
//   cmake --build .
// Executed as: main.exe
// And in Linux as follows:
//   cd Hw6
//   mkdir build && cd build
//   cmake ..
//   cmake --build .
// Executed as: ./main
// NOTE: If compilation fails, this link was extremely useful to verify the compiler:
// https://stackoverflow.com/questions/35869564/cmake-on-windows

#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#include <iostream>

constexpr int kArraySize = 16;

void OmpSectionsAndSingle4s2t(const float A[kArraySize], const float B[kArraySize], float C[kArraySize],
                          float D[kArraySize], float E[kArraySize], float F[kArraySize]) {
    printf("\n\n Method OmpSectionsAndSingle4s2t:\n");
#pragma omp parallel num_threads(4) shared(A, B, C, D, E, F)
    {
#pragma omp sections
        {
#pragma omp section
            {
                printf("C thread: %d\n", omp_get_thread_num());
                for (int i = 0; i < kArraySize; ++i) {
                    C[i] = A[i] + B[i];
                }
                Sleep(5);
            }  // End section.
#pragma omp section
            {
                printf("D thread: %d\n", omp_get_thread_num());
                for (int i = 0; i < kArraySize; ++i) {
                    D[i] = A[i] - B[i];
                }
                Sleep(5);
            }  // End section.
        }      // End sections.
#pragma omp single
        {
            printf("C= ");
            for (int i = 0; i < kArraySize; ++i) {
                printf("%.2f ", C[i]);
            }
            printf("\n");
        }  // End Single.
#pragma omp single
        {
            printf("D= ");
            for (int i = 0; i < kArraySize; ++i) {
                printf("%.2f ", D[i]);
            }
            printf("\n");
        }  // End Single.
#pragma omp single
        {
            printf("E= ");
            for (int i = 0; i < kArraySize; ++i) {
                printf("%.2f ", E[i]);
            }
            printf("\n");
        }  // End Single.
#pragma omp single
        {
            printf("F= ");
            for (int i = 0; i < kArraySize; ++i) {
                printf("%.2f ", F[i]);
            }
            printf("\n");
        }  // End Single.
    }      // End Parallel.
}

void OmpSectionsAndSingle2t4s(const float A[kArraySize], const float B[kArraySize], float C[kArraySize],
                          float D[kArraySize], float E[kArraySize], float F[kArraySize]) {
    printf("\n\n Method OmpSectionsAndSingle2t4s:\n");
#pragma omp parallel num_threads(2) shared(A, B, C, D, E, F)
    {
#pragma omp sections
        {
#pragma omp section
            {
                printf("C thread: %d\n", omp_get_thread_num());
                for (int i = 0; i < kArraySize; ++i) {
                    C[i] = A[i] + B[i];
                }
                Sleep(5);
            }  // End section.
#pragma omp section
            {
                printf("D thread: %d\n", omp_get_thread_num());
                for (int i = 0; i < kArraySize; ++i) {
                    D[i] = A[i] - B[i];
                }
                Sleep(5);
            }  // End section.
#pragma omp section
            {
                printf("E thread: %d\n", omp_get_thread_num());
                for (int i = 0; i < kArraySize; ++i) {
                    E[i] = A[i] * B[i];
                }
                Sleep(5);
            }  // End section.
#pragma omp section
            {
                printf("F thread: %d\n", omp_get_thread_num());
                for (int i = 0; i < kArraySize; ++i) {
                    F[i] = A[i] / B[i];
                }
                Sleep(5);
            }  // End section.
        }      // End sections.
#pragma omp single
        {
            printf("C= ");
            for (int i = 0; i < kArraySize; ++i) {
                printf("%.2f ", C[i]);
            }
            printf("\n");
        }  // End Single.
#pragma omp single
        {
            printf("D= ");
            for (int i = 0; i < kArraySize; ++i) {
                printf("%.2f ", D[i]);
            }
            printf("\n");
        }  // End Single.
#pragma omp single
        {
            printf("E= ");
            for (int i = 0; i < kArraySize; ++i) {
                printf("%.2f ", E[i]);
            }
            printf("\n");
        }  // End Single.
#pragma omp single
        {
            printf("F= ");
            for (int i = 0; i < kArraySize; ++i) {
                printf("%.2f ", F[i]);
            }
            printf("\n");
        }  // End Single.
    }      // End Parallel.
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

    for (int i = 0; i < kArraySize; ++i) {
        C[i] = 0;
        D[i] = 0;
        E[i] = 0;
        F[i] = 0;
    }

    OmpSectionsAndSingle2t4s(A, B, C, D, E, F);

    for (int i = 0; i < kArraySize; ++i) {
        C[i] = 0;
        D[i] = 0;
        E[i] = 0;
        F[i] = 0;
    }


    OmpSectionsAndSingle4s2t(A, B, C, D, E, F);

    return 0;
}