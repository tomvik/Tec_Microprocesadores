#include <MatrixMultiplier/SingleThreadMultiplier.h>

#include <string>
#include <vector>

namespace MatrixMultiplier {

SingleThreadMultiplier::SingleThreadMultiplier() : MatrixMultiplier() {}

void SingleThreadMultiplier::multiply(const std::vector<std::vector<double>>& matrix_a,
                                      const std::vector<std::vector<double>>& matrix_b,
                                      std::vector<std::vector<double>>* matrix_c) {
    for (int row_a = 0; row_a < matrix_a.size(); ++row_a) {
        for (int col_b = 0; col_b < matrix_b[0].size(); ++col_b) {
            (*matrix_c)[row_a][col_b] = 0;
            for (int col_a = 0; col_a < matrix_a[row_a].size(); ++col_a) {
                (*matrix_c)[row_a][col_b] += matrix_a[row_a][col_a] * matrix_b[col_a][col_b];
            }
        }
    }
}

}  // namespace MatrixMultiplier
