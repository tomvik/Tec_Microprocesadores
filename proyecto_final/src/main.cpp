// Code made by Tomas Alejandro Lugo Salinas
// for the Final project of the lecture of Multiprocessors.
// Compiled with Cmake in Windows as follows:
//   cd Hw6
//   mkdir build && cd build
//   cmake ..
//   cmake --build .
// Executed as: .\Debug\main.exe
// TODO(Tomas): check linux
// And in Linux as follows:
//   cd Hw6
//   mkdir build && cd build
//   cmake ..
//   cmake --build .
// Executed as: ./main
// NOTE: If compilation fails, this link was extremely useful to verify the compiler:
// https://stackoverflow.com/questions/35869564/cmake-on-windows

#include <ArgumentsCheck/ArgumentsCheck.h>
#include <GPUMatrix/GPUMatrix.h>
#include <MatrixCheck/MatrixCheck.h>
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

int main(int argc, char** argv) {
    ScopeTimer::ScopeTimer timer("Main function");

    std::vector<std::ifstream> input_files(2);

    const auto argument_case =
        ArgumentsCheck::handleArgumentsAndGetFileHandles(argc, argv, &input_files);
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

    std::vector<std::vector<double>> matrixA;
    std::vector<std::vector<double>> matrixB;

    const auto matrix_case = MatrixCheck::handleMatrixInput(&matrixA, &matrixB, &input_files);

    switch (matrix_case) {
    case MatrixCheck::MatrixCase::kOk:
        break;
    case MatrixCheck::MatrixCase::kNotEnoughLines:
        return 4;
    case MatrixCheck::MatrixCase::kNotEnoughMemory:
        return 5;
    case MatrixCheck::MatrixCase::kWrongDimensions:
        return 6;
    default:
        return 7;
    }

    /*for(int i = 0; i < 10; ++i) {
        std::string line;
        std::getline(input_files[0], line);
        std::cout << line << std::endl;
    }*/

    printMatrix(matrixA, "A");

    // GPUMatrix::HelloThreadIdx();
        return 0;
}