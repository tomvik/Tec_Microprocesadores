#include <MatrixMultiplier/CUDAMultiplier.h>

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
        const int matrix_a_rows = dimensions[0].first;
        const int matrix_a_cols = dimensions[0].second;
        const int matrix_b_cols = dimensions[1].second;

        for (int row_a = 0; row_a < matrix_a_rows; ++row_a)
        {
            for (int col_b = 0; col_b < matrix_b_cols; ++col_b)
            {
                double acum = 0;
                for (int col_a = 0; col_a < matrix_a_cols; ++col_a)
                {
                    acum += matrix_a[row_a][col_a] * matrix_b[col_a][col_b];
                }
                matrix_c[row_a][col_b] = acum;
            }
        }
    }

} // namespace MatrixMultiplier
