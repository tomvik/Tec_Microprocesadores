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

#include <iostream>

int main(int argc, char** argv) {
    ScopeTimer::ScopeTimer timer("Main function");
    
    const auto argument_case = ArgumentsCheck::handleArguments(argc, argv);
    switch (argument_case) {
        case ArgumentsCheck::ArgumentsCase::kHelp:
            std::cout << "kHelp" << std::endl;
            return 0;
        case ArgumentsCheck::ArgumentsCase::kWrongArguments:
            std::cout << "kWrongArguments" << std::endl;
            return 1;
        case ArgumentsCheck::ArgumentsCase::kOk:
            std::cout << "kOk" << std::endl;
        default:
            break;
    }

    GPUMatrix::HelloThreadIdx();
    return 0;
}