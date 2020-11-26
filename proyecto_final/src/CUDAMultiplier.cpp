#include <MatrixMultiplier/CUDAMultiplier.h>
#include <GPUMatrix/GPUMatrix.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace MatrixMultiplier
{

    CUDAMultiplier::CUDAMultiplier()
        : MatrixMultiplier("CUDA", 1024) {}

    void CUDAMultiplier::multiply(double **matrix_a, double **matrix_b, double **matrix_c,
                                  const std::vector<std::pair<int, int>> &dimensions)
    {
        GPUMatrix::CUDAMultiplier(matrix_a, matrix_b, matrix_c, dimensions);
    }

} // namespace MatrixMultiplier
