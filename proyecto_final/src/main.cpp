// Code made by Tomas Alejandro Lugo Salinas
// for the Final project of the lecture of Multiprocessors.
// Compiled with Cmake in Windows as follows:
//   cd Hw6
//   mkdir build && cd build
//   cmake ..
//   cmake --build .
// Executed as: .\Debug\main.exe "Path_to_MatrixA.txt" "Path_to_MatrixB.txt"
// TODO(Tomas): check linux
// And in Linux as follows:
//   cd Hw6
//   mkdir build && cd build
//   cmake ..
//   cmake --build .
// Executed as: ./main "Path_to_MatrixA.txt" "Path_to_MatrixB.txt"
// NOTE: If compilation fails, this link was extremely useful to verify the compiler:
// https://stackoverflow.com/questions/35869564/cmake-on-windows

#include <ArgumentsCheck/ArgumentsCheck.h>
//#include <GPUMatrix/GPUMatrix.h>
#include <MatrixCheck/MatrixCheck.h>
#include <MatrixMultiplier/MatrixMultiplier.h>
#include <MatrixMultiplier/OMPMultiplier.h>
#include <MatrixMultiplier/PThreadMultiplier.h>
#include <MatrixMultiplier/SingleThreadMultiplier.h>
#include <ScopeTimer/ScopeTimer.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <vector>

template <typename T>
void printMatrix(const std::vector<std::vector<T>>& matrix, const std::string name) {
    std::cout << "Printing matrix: " << name << "\n";
    for (const auto& row : matrix) {
        for (const auto& element : row) {
            std::cout << element << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    std::cout << std::endl;
}

void printMatrix(double** matrix, const std::pair<int, int> dimensions, const std::string name) {
    std::cout << "Printing matrix: " << name << "\n";
    for (int i = 0; i < dimensions.first; ++i) {
        for (int j = 0; j < dimensions.second; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    const ScopeTimer::ScopeTimer timer("Main function");

    std::vector<std::ifstream> input_files(2);
    std::string output_file_path = "";

    const auto& argument_case = ArgumentsCheck::handleArgumentsAndGetFileHandles(
        argc, argv, &input_files, &output_file_path);
    switch (argument_case) {
        case ArgumentsCheck::ArgumentsCase::kHelp:
            return 0;
        case ArgumentsCheck::ArgumentsCase::kWrongArguments:
            return 1;
        case ArgumentsCheck::ArgumentsCase::kWrongPathOrFile:
            return 2;
        case ArgumentsCheck::ArgumentsCase::kOk:
            break;
        default:
            return 3;
    }

    std::vector<std::pair<int, int>> dimensions;

    const auto& matrix_input_case = MatrixCheck::handleMatrixInput(&dimensions);

    switch (matrix_input_case) {
        case MatrixCheck::MatrixCase::kOk:
            break;
        case MatrixCheck::MatrixCase::kWrongDimensions:
            return 4;
        default:
            return 5;
    }

    const int64_t len_a = sizeof(double*) * dimensions[0].first +
                          sizeof(double) * dimensions[0].second * dimensions[0].first;
    const int64_t len_b = sizeof(double*) * dimensions[1].first +
                          sizeof(double) * dimensions[1].second * dimensions[1].first;
    const int64_t len_c = sizeof(double*) * dimensions[2].first +
                          sizeof(double) * dimensions[2].second * dimensions[2].first;

    double** matrix_a = reinterpret_cast<double**>(malloc(len_a));
    double** matrix_b = reinterpret_cast<double**>(malloc(len_b));
    double** matrix_c = reinterpret_cast<double**>(malloc(len_c));

    const auto& matrix_malloc_case =
        MatrixCheck::handleMallocAndFilling(matrix_a, matrix_b, matrix_c, &input_files, dimensions);

    switch (matrix_malloc_case) {
        case MatrixCheck::MatrixCase::kOk:
            break;
        case MatrixCheck::MatrixCase::kNotEnoughMemory:
            return 6;
        case MatrixCheck::MatrixCase::kNotEnoughLines:
            return 7;
        default:
            return 8;
    }

    std::vector<MatrixMultiplier::MatrixMultiplier*> multipliers;

    multipliers.emplace_back(new MatrixMultiplier::SingleThreadMultiplier());
    multipliers.emplace_back(new MatrixMultiplier::OMPMultiplier(16));
    multipliers.emplace_back(new MatrixMultiplier::PThreadMultiplier(16));

    for (int i = 0; i < multipliers.size(); ++i) {
        multipliers[i]->multiplyNTimes(matrix_a, matrix_b, matrix_c, dimensions, 5,
                                       output_file_path);
    }

    free(matrix_a);
    free(matrix_b);
    free(matrix_c);

    return 0;
}