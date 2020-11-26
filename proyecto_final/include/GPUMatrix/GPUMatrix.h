
#ifndef PROYECTO_FINAL_INCLUDE_GPUMATRIX_GPUMATRIX_H_
#define PROYECTO_FINAL_INCLUDE_GPUMATRIX_GPUMATRIX_H_

namespace GPUMatrix
{

    void CUDAMultiplier(double **matrix_a, double **matrix_b, double **matrix_c,
                        const std::vector<std::pair<int, int>> &dimensions);

} // namespace GPUMatrix

#endif // PROYECTO_FINAL_INCLUDE_GPUMATRIX_GPUMATRIX_H_
