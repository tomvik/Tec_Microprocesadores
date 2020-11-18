
#ifndef PROYECTO_FINAL_INCLUDE_MATRIXMULTIPLIER_SINGLETHREADMULTIPLIER_H_
#define PROYECTO_FINAL_INCLUDE_MATRIXMULTIPLIER_SINGLETHREADMULTIPLIER_H_

#include <MatrixMultiplier/MatrixMultiplier.h>

#include <string>
#include <vector>

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
    void multiply(const std::vector<std::vector<double>>& matrix_a,
                          const std::vector<std::vector<double>>& matrix_b,
                          std::vector<std::vector<double>>* matrix_c);
};

}  // namespace MatrixMultiplier

#endif  // PROYECTO_FINAL_INCLUDE_MATRIXMULTIPLIER_SINGLETHREADMULTIPLIER_H_
