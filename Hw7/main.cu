// Code made by Tomas Alejandro Lugo Salinas
// for the Hw7 of the lecture of Multiprocessors.
// Compiled in Windows 10 with: nvcc -Xcompiler "/openmp" main.cu -o main.exe
// Executed in Windows 10 as: main.exe

#include <stdio.h>
#include <time.h>
#include <omp.h>

const long kCantidadIntervalos = 1000000000;

void originalPi(const long cantidad_intervalos, const int times = 1) {
    printf("** Running the original code %d times **\n", times);
    long total_time = 0;
    double baseIntervalo;
    double fdx;
    double acum = 0;
    clock_t start, end;
    for(int iteration = 0; iteration < times; ++iteration) {
        double x = 0;
        long i;
        baseIntervalo = 1.0 / cantidad_intervalos;
        start = clock();
        for (i = 0; i < cantidad_intervalos; i++) {
           x = (i+0.5)*baseIntervalo;
           fdx = 4 / (1 + x * x);
           acum += fdx;
        }
        acum *= baseIntervalo;
        end = clock();
        total_time += (end - start);
        printf("Result = %20.18lf (%ld)\n", acum, end - start);
    }
    printf("** The average of %d runs was: %ld **\n\n", times, total_time / times);
}

/////////////////// [ Begins code inspired from the internet ] //////////////////////
// The link of the reduction function guide is: https://riptutorial.com/cuda/example/22460/single-warp-parallel-reduction-for-commutative-operator
// Also, please note that a floating addition is not associative, so it gives some errors: https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html

static const int warpSize = 32;
static const int blockSize = 1024;

__device__ int sumCommSingleWarp(volatile double* shArr) {
    const int idx = threadIdx.x % warpSize; // The lane index in the warp
    if (idx < 16) {
      shArr[idx] += shArr[idx + 16];
      shArr[idx] += shArr[idx + 8];
      shArr[idx] += shArr[idx + 4];
      shArr[idx] += shArr[idx + 2];
      shArr[idx] += shArr[idx + 1];
    }
    return shArr[0];
}

__global__ void singleReduction(const int arraySize, const double *a, double *out) {
    const int idx = threadIdx.x;
    double sum = 0;
    for (int i = idx; i < arraySize; i += blockSize)
        sum += a[i];
    __shared__ double r[blockSize];
    r[idx] = sum;
    sumCommSingleWarp(&r[idx & ~(warpSize-1)]);
    __syncthreads();
    if (idx<warpSize) { //first warp only
        r[idx] = idx * warpSize < blockSize ? r[idx*warpSize] : 0;
        sumCommSingleWarp(r);
        if (idx == 0)
            *out = r[0];
    }
}
/////////////////// [ Ends code inspired from the internet ] //////////////////////

__global__ void singleGPUPi(const long cantidad_intervalos, const int total_threads, const int thread_per_block, double *acum_arr) {
    const double base_intervalo = 1.0 / cantidad_intervalos;
    const int idx = threadIdx.x + (blockIdx.x * thread_per_block);

    const long intervalos_local = cantidad_intervalos / total_threads;
    double x = base_intervalo * intervalos_local * idx;
    double fdx;
    double local_acum = 0;

    for (long i = 0; i < intervalos_local; i++) {
        fdx = 4.0 / (1 + x * x);
        local_acum += (fdx * base_intervalo);
        x = x + base_intervalo;
    }

    acum_arr[idx] = local_acum;
}

void gpuPiWithReduction(const long cantidad_intervalos, const int num_blocks, const int num_threads, const int times = 1) {
    printf("** Running the gpu code with reduction %d times **\n", times);
    printf("* # of blocks: %d\n", num_blocks);
    printf("* # of threads: %d\n", num_threads);
    long total_time = 0;
    clock_t start, end;
    for(int iteration = 0; iteration < times; ++iteration) {
        start = clock();

        const int total_size = num_blocks * num_threads;

        double *acum_arr = nullptr;

        cudaMallocManaged(&acum_arr, sizeof(double) * total_size);
        
        singleGPUPi<<<num_blocks, num_threads>>>(cantidad_intervalos, total_size, num_threads, acum_arr);

        cudaDeviceSynchronize();

        double *acum = nullptr;
        double final_acum = 0;

        cudaMallocManaged(&acum, sizeof(double));

        singleReduction<<<1, num_threads>>>(total_size, acum_arr, acum);

        cudaDeviceSynchronize();

        final_acum = *acum;

        cudaFree(acum_arr);
        cudaFree(acum);

        end = clock();
        total_time += (end - start);
        printf("Result = %20.18lf (%ld)\n", final_acum, end - start);
    }
    printf("** The average of %d runs was: %ld **\n\n", times, total_time / times);
}

void gpuPiWithoutReduction(const long cantidad_intervalos, const int num_blocks, const int num_threads, const int times = 1) {
    printf("** Running the gpu code without reduction %d times **\n", times);
    printf("* # of blocks: %d\n", num_blocks);
    printf("* # of threads: %d\n", num_threads);
    long total_time = 0;
    clock_t start, end;
    for(int iteration = 0; iteration < times; ++iteration) {
        start = clock();

        const int total_size = num_blocks * num_threads;

        double *acum_arr = nullptr;

        cudaMallocManaged(&acum_arr, sizeof(double) * total_size);
        
        singleGPUPi<<<num_blocks, num_threads>>>(cantidad_intervalos, total_size, num_threads, acum_arr);

        cudaDeviceSynchronize();
        
        double final_acum = 0;

        for(int i = 0; i < total_size; ++i) {
            final_acum += acum_arr[i];
        }

        cudaFree(acum_arr);

        end = clock();
        total_time += (end - start);
        printf("Result = %20.18lf (%ld)\n", final_acum, end - start);
    }
    printf("** The average of %d runs was: %ld **\n\n", times, total_time / times);
}

void gpuPiWithOMPReduction(const long cantidad_intervalos, const int num_blocks, const int num_threads, const int times = 1) {
    printf("** Running the gpu code with OMP reduction %d times **\n", times);
    printf("* # of blocks: %d\n", num_blocks);
    printf("* # of threads: %d\n", num_threads);
    long total_time = 0;
    clock_t start, end;
    for(int iteration = 0; iteration < times; ++iteration) {
        start = clock();

        const int total_size = num_blocks * num_threads;

        double *acum_arr = nullptr;

        cudaMallocManaged(&acum_arr, sizeof(double) * total_size);
        
        singleGPUPi<<<num_blocks, num_threads>>>(cantidad_intervalos, total_size, num_threads, acum_arr);

        cudaDeviceSynchronize();
        
        double final_acum = 0;

        #pragma omp parallel for reduction(+:final_acum)
        for(int i = 0; i < total_size; ++i) {
            final_acum += acum_arr[i];
        }

        cudaFree(acum_arr);

        end = clock();
        total_time += (end - start);
        printf("Result = %20.18lf (%ld)\n", final_acum, end - start);
    }
    printf("** The average of %d runs was: %ld **\n\n", times, total_time / times);
}

int main() {
    //originalPi(kCantidadIntervalos, 5);
    gpuPiWithReduction(kCantidadIntervalos, 32, 1024, 5);
    gpuPiWithoutReduction(kCantidadIntervalos, 32, 1024, 5);
    gpuPiWithOMPReduction(kCantidadIntervalos, 32, 1024, 5);
    return 0;
}