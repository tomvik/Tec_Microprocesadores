#include <stdio.h>

__global__ 
void cuda_hello() { printf("Hello World from GPU! %d\n", threadIdx.x); }

int main() {
    cuda_hello<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
