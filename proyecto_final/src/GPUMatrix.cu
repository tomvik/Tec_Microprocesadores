#include <GPUMatrix/GPUMatrix.h>
#include <stdio.h>
#include <vector>
#include <utility> 
namespace {
    __global__ void mul(double* matrix_a, double* matrix_b, double* matrix_c, int m, int n, int colA) {
        // printf("Hello World from GPU! %d %d\n", blockIdx.x, threadIdx.x);
        double suma = 0.0;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if((row < m) && (col < n)) {
            for(int i = 0; i < colA; i++) {
                suma += matrix_a[colA * row + i] + 3 * matrix_b[col + i * n];
            }
            matrix_c[row * n + col] = suma;
        }
    }
}

namespace GPUMatrix {
    
void CUDAMultiplier(double **matrix_a, double **matrix_b, double **matrix_c, const std::vector<std::pair<int, int> > &dimensions) {
    const int64_t len_a = sizeof(double *) * dimensions[0].first +
                            sizeof(double) * dimensions[0].second * dimensions[0].first;
    const int64_t len_b = sizeof(double *) * dimensions[1].first +
                          sizeof(double) * dimensions[1].second * dimensions[1].first;
    const int64_t len_c = sizeof(double *) * dimensions[2].first +
                          sizeof(double) * dimensions[2].second * dimensions[2].first;

                          double* matrix_a_device;
                          double* matrix_b_device;
    double* matrix_c_device;


    cudaMalloc((void**)& matrix_a_device, len_a);
    cudaMalloc((void**)& matrix_b_device, len_b);
    cudaMalloc((void**)& matrix_c_device, len_c);
    
    cudaMemcpy(matrix_a_device, matrix_a, len_a, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_b_device, matrix_b, len_b, cudaMemcpyHostToDevice);
    
    cudaMemset(matrix_c_device, 0, len_c);
    
    dim3 dimBlock(1,1);
    dim3 dimGrid(dimensions[1].second, dimensions[1].first);
    
    mul<<<dimGrid, dimBlock>>>(matrix_a_device, matrix_b_device, matrix_c_device, dimensions[0].first, dimensions[1].first, dimensions[1].second); 
    cudaDeviceSynchronize();

    cudaMemcpy(matrix_c_device, matrix_c, len_c, cudaMemcpyDeviceToHost);

    cudaFree(matrix_a_device);
    cudaFree(matrix_b_device);
    cudaFree(matrix_c_device);
    }

}  // namespace GPUMatrix