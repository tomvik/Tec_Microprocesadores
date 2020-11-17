#ifndef PROYECTO_FINAL_INCLUDE_MATRIXCHECK_MATRIXCHECK_H_
#define PROYECTO_FINAL_INCLUDE_MATRIXCHECK_MATRIXCHECK_H_

#include <fstream>
#include <vector>

namespace MatrixCheck {

// Enum class for the possible matrix cases.
enum class MatrixCase { kOk = 0, kWrongDimensions = 1, kNotEnoughMemory = 2, kNotEnoughLines = 3 };

// Checks if the Matrices are correct, and prints the appropiate message if neccesary.
// It also updates the value of the variables real_dimension and real_num_threads.

MatrixCase handleMatrixInput(std::vector<std::vector<double>>* matrix_a,
                             std::vector<std::vector<double>>* matrix_b,
                             std::vector<std::ifstream>* input_files);

}  // namespace MatrixCheck

#endif  // PROYECTO_FINAL_INCLUDE_MATRIXCHECK_MATRIXCHECK_H_
