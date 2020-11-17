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

bool fillMatrix(std::vector<std::vector<double>>* matrix, std::vector<std::ifstream>* input_files,
                const int file_number) {
    int counter = 0;
    const char letter = 'A' + file_number;

    for (int i = 0; i < (*matrix).size(); ++i) {
        for (int j = 0; j < (*matrix)[i].size(); ++j) {
            std::string line;
            if (!std::getline((*input_files)[0], line)) {
                std::cerr << "[ ERROR ] "
                          << "Not enough lines in file " << letter << ".\n"
                          << "          "
                          << "The number of lines in the file is:\n"
                          << "          " << counter << std::endl;
                return false;
            }
            ++counter;
            std::istringstream iss(line);
            iss >> (*matrix)[i][j];
        }
    }
}

}  // namespace

namespace MatrixCheck {

MatrixCase handleMatrixInput(std::vector<std::vector<double>>* matrix_a,
                             std::vector<std::vector<double>>* matrix_b,
                             std::vector<std::ifstream>* input_files) {
    const auto& dimensions = getDimensions();

    const int matrix_a_rows = dimensions[0].first;
    const int matrix_a_cols = dimensions[0].second;
    const int matrix_b_rows = dimensions[1].first;
    const int matrix_b_cols = dimensions[1].second;

    if (matrix_a_cols != matrix_b_rows) {
        std::cerr
            << "[ ERROR ] "
            << "The sizes are incompatible. The columns of A must be the same as the rows of B."
            << std::endl;
        return MatrixCase::kWrongDimensions;
    }

    try {
        std::vector<std::vector<double>> matrixA(matrix_a_rows, std::vector<double>(matrix_a_cols));
        std::vector<std::vector<double>> matrixB(matrix_b_rows, std::vector<double>(matrix_b_cols));

        *matrix_a = matrixA;
        *matrix_b = matrixB;
    } catch (const std::bad_alloc& ba) {
        std::cerr << "[ ERROR ] "
                  << "Not enough memory" << std::endl;
        return MatrixCase::kNotEnoughMemory;
    }

    if (fillMatrix(matrix_a, input_files, 0) && fillMatrix(matrix_b, input_files, 1)) {
        return MatrixCase::kOk;
    }

    return MatrixCase::kNotEnoughLines;
}

}  // namespace MatrixCheck
