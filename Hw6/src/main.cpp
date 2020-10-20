// Code made by Tomas Alejandro Lugo Salinas
// for the Hw6 of the lecture of Multiprocessors.
// Compiled with Cmake in Windows as follows:
//   cd Hw6
//   mkdir build && cd build
//   cmake .. -G"MinGW Makefiles"
//   cmake --build .
// And in Linux as follows:
//   cd Hw6
//   mkdir build && cd build
//   cmake ..
//   cmake --build .
// Executed as: main.exe
// NOTE: If compilation fails, this link was extremely useful to verify the compiler:
// https://stackoverflow.com/questions/35869564/cmake-on-windows

#include <Timer/Timer.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

constexpr int kArraySize = 16;

void Sequential(const float A[kArraySize], const float B[kArraySize], float C[kArraySize],
                float D[kArraySize], float E[kArraySize], float F[kArraySize]) {
    printf("\n\n Method Sequential:\n");
    Timer::Timer timer("Sequential");
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

void OmpSectionsAndSingle(const float A[kArraySize], const float B[kArraySize], float C[kArraySize],
                          float D[kArraySize], float E[kArraySize], float F[kArraySize]) {
    printf("\n\n Method OmpSectionsAndSingle:\n");
    Timer::Timer timer("OmpSectionsAndSingle");
#pragma omp parallel num_threads(4) shared(A, B, C, D, E, F)
    {
#pragma omp sections
        {
#pragma omp section
            {
                // printf("C thread: %d\n", omp_get_thread_num());
                for (int i = 0; i < kArraySize; ++i) {
                    C[i] = A[i] + B[i];
                }
            }  // End section.
#pragma omp section
            {
                // printf("D thread: %d\n", omp_get_thread_num());
                for (int i = 0; i < kArraySize; ++i) {
                    D[i] = A[i] - B[i];
                }
            }  // End section.
#pragma omp section
            {
                // printf("E thread: %d\n", omp_get_thread_num());
                for (int i = 0; i < kArraySize; ++i) {
                    E[i] = A[i] * B[i];
                }
            }  // End section.
#pragma omp section
            {
                // printf("F thread: %d\n", omp_get_thread_num());
                for (int i = 0; i < kArraySize; ++i) {
                    F[i] = A[i] / B[i];
                }
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

void OmpSectionsAndCritical(const float A[kArraySize], const float B[kArraySize],
                            float C[kArraySize], float D[kArraySize], float E[kArraySize],
                            float F[kArraySize]) {
    printf("\n\n Method OmpSectionsAndCritical:\n");
    Timer::Timer timer("OmpSectionsAndCritical");
#pragma omp parallel num_threads(4) shared(A, B, C, D, E, F)
    {
#pragma omp sections
        {
#pragma omp section
            {
                // printf("C thread: %d\n", omp_get_thread_num());
                for (int i = 0; i < kArraySize; ++i) {
                    C[i] = A[i] + B[i];
                }
            }  // End section.
#pragma omp section
            {
                // printf("D thread: %d\n", omp_get_thread_num());
                for (int i = 0; i < kArraySize; ++i) {
                    D[i] = A[i] - B[i];
                }
            }  // End section.
#pragma omp section
            {
                // printf("E thread: %d\n", omp_get_thread_num());
                for (int i = 0; i < kArraySize; ++i) {
                    E[i] = A[i] * B[i];
                }
            }  // End section.
#pragma omp section
            {
                // printf("F thread: %d\n", omp_get_thread_num());
                for (int i = 0; i < kArraySize; ++i) {
                    F[i] = A[i] / B[i];
                }
            }  // End section.
        }      // End sections.
#pragma omp critical
        {
            const int thread_num = omp_get_thread_num();
            printf("%c= ", 'C' + thread_num);
            switch (thread_num) {
                case 0:
                    for (int i = 0; i < kArraySize; ++i) {
                        printf("%.2f ", C[i]);
                    }
                    break;
                case 1:
                    for (int i = 0; i < kArraySize; ++i) {
                        printf("%.2f ", D[i]);
                    }
                    break;
                case 2:
                    for (int i = 0; i < kArraySize; ++i) {
                        printf("%.2f ", E[i]);
                    }
                    break;
                case 3:
                    for (int i = 0; i < kArraySize; ++i) {
                        printf("%.2f ", F[i]);
                    }
                    break;
                default:
                    break;
            }
            printf("\n");
        }  // End Critical
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

    Sequential(A, B, C, D, E, F);

    for (int i = 0; i < kArraySize; ++i) {
        C[i] = 0;
        D[i] = 0;
        E[i] = 0;
        F[i] = 0;
    }

    OmpSectionsAndSingle(A, B, C, D, E, F);

    for (int i = 0; i < kArraySize; ++i) {
        C[i] = 0;
        D[i] = 0;
        E[i] = 0;
        F[i] = 0;
    }

    OmpSectionsAndCritical(A, B, C, D, E, F);

    return 0;
}