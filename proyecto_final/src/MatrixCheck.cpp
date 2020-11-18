#include <MatrixCheck/MatrixCheck.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

std::vector<std::pair<int, int>> getDimensions() {
    std::vector<std::pair<int, int>> dimensions(2);

    for (int i = 0; i < 2; ++i) {
        const char letter = 'A' + i;

        std::cout << "What would the size of the Matrix " << letter << " be?" << std::endl;
        std::cout << "Matrix " << letter << " rows: ";
        std::cin >> dimensions[i].first;
        std::cout << "Matrix " << letter << " columns: ";
        std::cin >> dimensions[i].second;
    }

    for (int i = 0; i < 2; ++i) {
        const char letter = 'A' + i;

        std::cout << "Matrix " << letter << " will be: " << dimensions[i].first << "x"
                  << dimensions[i].second << std::endl;
    }

    return dimensions;
}

bool fillMatrix(double** matrix, const int rows, const int cols,
                std::vector<std::ifstream>* input_files, const int file_number) {
    int counter = 0;
    const char letter = 'A' + file_number;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::string line;
            if (!std::getline((*input_files)[file_number], line)) {
                std::cerr << "[ ERROR ] "
                          << "Not enough lines in file " << letter << ".\n"
                          << "          "
                          << "The number of lines in the file is:\n"
                          << "          " << counter << std::endl;
                return false;
            }
            ++counter;
            std::istringstream iss(line);
            iss >> matrix[i][j];
        }
    }
    return true;
}

}  // namespace

namespace MatrixCheck {

MatrixCase handleMatrixInput(std::vector<std::pair<int, int>>* dimensions) {
    *dimensions = getDimensions();

    const int matrix_a_rows = (*dimensions)[0].first;
    const int matrix_a_cols = (*dimensions)[0].second;
    const int matrix_b_rows = (*dimensions)[1].first;
    const int matrix_b_cols = (*dimensions)[1].second;

    if (matrix_a_cols != matrix_b_rows ||
        (matrix_a_rows <= 0 || matrix_a_cols <= 0 || matrix_b_rows <= 0 || matrix_b_cols <= 0)) {
        std::cerr
            << "[ ERROR ] "
            << "The sizes are incompatible. The columns of A must be the same as the rows of B.\n"
            << "          " << "And all sizes must be positive greater than 0."
            << std::endl;
        return MatrixCase::kWrongDimensions;
    }

    std::pair<int, int> c_dim = {matrix_a_rows, matrix_b_cols};

    (*dimensions).emplace_back(c_dim);
    return MatrixCase::kOk;
}

MatrixCase handleMallocAndFilling(double** matrix_a, double** matrix_b, double** matrix_c,
                                  std::vector<std::ifstream>* input_files,
                                  const std::vector<std::pair<int, int>>& dimensions) {
    if (matrix_a == nullptr || matrix_b == nullptr || matrix_c == nullptr) {
        std::cerr << "[ ERROR ] "
                  << "Not enough memory\n"
                  << "          " << std::endl;
        return MatrixCase::kNotEnoughMemory;
    }

    double* ptr = (double*)(matrix_a + dimensions[0].first);
    for (int i = 0; i < dimensions[0].first; i++) matrix_a[i] = (ptr + dimensions[0].second * i);

    ptr = (double*)(matrix_b + dimensions[1].first);
    for (int i = 0; i < dimensions[1].first; i++) matrix_b[i] = (ptr + dimensions[1].second * i);

    ptr = (double*)(matrix_c + dimensions[2].first);
    for (int i = 0; i < dimensions[2].first; i++) matrix_c[i] = (ptr + dimensions[2].second * i);

    if (fillMatrix(matrix_a, dimensions[0].first, dimensions[0].second, input_files, 0) &&
        fillMatrix(matrix_b, dimensions[1].first, dimensions[1].second, input_files, 1)) {
        return MatrixCase::kOk;
    }

    return MatrixCase::kNotEnoughLines;
}

}  // namespace MatrixCheck
