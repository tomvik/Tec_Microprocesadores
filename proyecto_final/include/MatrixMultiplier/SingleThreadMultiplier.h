
#ifndef PROYECTO_FINAL_INCLUDE_MATRIXMULTIPLIER_SINGLETHREADMULTIPLIER_H_
#define PROYECTO_FINAL_INCLUDE_MATRIXMULTIPLIER_SINGLETHREADMULTIPLIER_H_

#include <MatrixMultiplier/MatrixMultiplier.h>

#include <string>
#include <vector>
#include <utility>

namespace MatrixMultiplier {

class SingleThreadMultiplier : public MatrixMultiplier {
 public:
    // Creates a SingleThreadMultiplier.
    SingleThreadMultiplier();

    // Destructor of the SingleThreadMultiplier.
    ~SingleThreadMultiplier();

 protected:
    // Multiplies matrix_a and matrix_b storing it in matrix_c
    // By this point it is assured that the matrices are correct and can be multiplied.
    void multiply(double** matrix_a, double** matrix_b, double** matrix_c,
                  const std::vector<std::pair<int, int>>& dimensions);
};

}  // namespace MatrixMultiplier

#endif  // PROYECTO_FINAL_INCLUDE_MATRIXMULTIPLIER_SINGLETHREADMULTIPLIER_H_