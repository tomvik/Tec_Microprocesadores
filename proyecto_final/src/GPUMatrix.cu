#include <GPUMatrix/GPUMatrix.h>
#include <stdio.h>

namespace {
    __global__
    void mul() {
    // void mul(double* A, double* B, double* A, int m, int n, int* colA) {
        printf("Hello World from GPU! %d %d\n", blockIdx.x, threadIdx.x);
        // int row = blockIdx.y * blockDim.y + threadIdx.y;
        // int col = blockIdx.x * blockDim.x + threadIdx.x;

        // double sum = 0.0;
        // if((row < m) && (col < n)) {
        //     for(int i = 0; i < *colA; i++) {
        //         sum += A[*colA * row + i] * B[col + i * n];
        //     }
        //     C[row * n + col] = sum;
        // }
    }
}

namespace GPUMatrix {
    
void HelloThreadIdx() { mul<<<2, 4>>>(); }

}  // namespace GPUMatrix