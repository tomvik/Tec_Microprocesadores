#include <stdio.h>

__global__ 
void HelloThreadIdx() { printf("Hello World from GPU! %d %d\n", blockIdx.x, threadIdx.x); }

__global__ 
void HelloBlockThreadIdx() { printf("Hello World from GPU! %d %d\n", blockIdx.x, threadIdx.x); }

int main() {
    HelloThreadIdx<<<2, 4>>>();
    HelloBlockThreadIdx<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
