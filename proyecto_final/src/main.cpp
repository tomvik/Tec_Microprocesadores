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
#include <ScopeTimer/ScopeTimer.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <vector>

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
        default:
            break;
    }

    GPUMatrix::HelloThreadIdx();
    return 0;
}