#include <MatrixMultiplier/PThreadMultiplier.h>
#include <omp.h>
#include <pthread.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace MatrixMultiplier {

PThreadMultiplier::PThreadMultiplier(const int amount_of_threads)
    : MatrixMultiplier(), amount_of_threads_(amount_of_threads) {}

void PThreadMultiplier::multiply(double** matrix_a, double** matrix_b, double** matrix_c,
                                 const std::vector<std::pair<int, int>>& dimensions) {
    const int matrix_a_rows = dimensions[0].first;
    const int matrix_a_cols = dimensions[0].second;
    const int matrix_b_cols = dimensions[1].second;

#pragma omp parallel for num_threads(amount_of_threads_)
    for (int row_a = 0; row_a < matrix_a_rows; ++row_a) {
        for (int col_b = 0; col_b < matrix_b_cols; ++col_b) {
            matrix_c[row_a][col_b] = 0;
            for (int col_a = 0; col_a < matrix_a_cols; ++col_a) {
                matrix_c[row_a][col_b] += matrix_a[row_a][col_a] * matrix_b[col_a][col_b];
            }
        }
    }
}

}  // namespace MatrixMultiplier
