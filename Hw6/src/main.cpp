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

int main() {
    const float A[16] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    const float B[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float C[16] = {0};
    float D[16] = {0};
    float E[16] = {0};
    float F[16] = {0};

    Timer::Timer timer("Sequencial");
    for (int i = 0; i < 16; ++i) {
        C[i] = A[i] + B[i];
        D[i] = A[i] - B[i];
        E[i] = A[i] * B[i];
        F[i] = A[i] / B[i];
    }

    return 0;
}