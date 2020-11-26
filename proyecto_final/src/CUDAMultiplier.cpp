#include <MatrixMultiplier/CUDAMultiplier.h>
#include <GPUMatrix/GPUMatrix.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace MatrixMultiplier
{

    CUDAMultiplier::CUDAMultiplier(const int amount_of_threads)
        : MatrixMultiplier("CUDA", amount_of_threads) {}

    void CUDAMultiplier::multiply(double **matrix_a, double **matrix_b, double **matrix_c,
                                  const std::vector<std::pair<int, int>> &dimensions)
    {
        GPUMatrix::CUDAMultiplier(matrix_a, matrix_b, matrix_c, dimensions);
    }

} // namespace MatrixMultiplier
