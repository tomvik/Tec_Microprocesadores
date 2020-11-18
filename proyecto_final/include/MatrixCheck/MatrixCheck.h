#ifndef PROYECTO_FINAL_INCLUDE_MATRIXCHECK_MATRIXCHECK_H_
#define PROYECTO_FINAL_INCLUDE_MATRIXCHECK_MATRIXCHECK_H_

#include <fstream>
#include <utility>
#include <vector>

namespace MatrixCheck {

// Enum class for the possible matrix cases.
enum class MatrixCase { kOk = 0, kWrongDimensions = 1, kNotEnoughMemory = 2, kNotEnoughLines = 3 };

// Checks if the Matrices dimensions are correct, and prints the appropiate error message if
// neccesary. It updates the dimensions vector with the dimensions of the A and B matrix.
MatrixCase handleMatrixInput(std::vector<std::pair<int, int>>* dimensions);

// Checks if the Malloc worked and also if it can correctly fill the a and b matrices.
// It updates such matrices and in case of an error it prints our the appropriate error message.
MatrixCase handleMallocAndFilling(double** matrix_a, double** matrix_b, double** matrix_c,
                                  std::vector<std::ifstream>* input_files,
                                  const std::vector<std::pair<int, int>>& dimensions);

}  // namespace MatrixCheck

#endif  // PROYECTO_FINAL_INCLUDE_MATRIXCHECK_MATRIXCHECK_H_
