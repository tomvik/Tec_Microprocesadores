#include <stdio.h>

__global__ 
void cuda_hello() { printf("Hello World from GPU!\n"); }

int main() {
    cuda_hello<<<10, 1>>>();
    return 0;
}
