#include "ArgumentsCheck.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace ArgumentsCheck {

namespace {

constexpr int kDimArg = 1;
constexpr int kThreadArg = 2;

bool isNotDigit(const unsigned char c) { return !std::isdigit(c); }

bool isPositiveNumber(const std::string& s) {
    return std::find_if(s.begin(), s.end(), isNotDigit) == s.end();
}

void printOkMessage(const int dimension, const int num_threads) {
    printf("The dimensions are: %dx%d, and the amount of threads is: %d\n", dimension, dimension,
           num_threads);
}

void printHelpMessage(char** argv) {
    std::cerr << "Usage: " << argv[0] << " {<option>} or {<DIMENSION> <THREADS>}\n"
              << "Option:\n"
              << "\t-h,--help\t\tShow this help message\n"
              << "DIMENSION:\n"
              << "\tDimension of the three square matrices, must be greater than 0\n"
              << "THREADS:\n"
              << "\tNumber of threads to execute the program with, must be greater than 0\n"
              << std::endl;
}

void printWrongArguments(int argc, char** argv) {
    std::cerr << "The following execution is incorrect: ";
    for (int i = 0; i < argc; ++i) {
        std::cerr << argv[i] << " ";
    }
    std::cerr << "\n"
              << "Run the following to check the correct execution:\n"
              << "\t" << argv[0] << " -h" << std::endl;
}

}  // namespace

ArgumentsCase handleArguments(int* real_dimension, int* real_num_threads, int argc, char** argv) {
    std::vector<std::string> arguments;
    for (int i = 0; i < argc; ++i) {
        arguments.push_back(argv[i]);
    }
    if (argc > 1) {
        if (arguments[1] == "-h" || arguments[1] == "--help") {
            printHelpMessage(argv);
            return ArgumentsCase::kHelp;
        } else if (argc == 3) {
            if (isPositiveNumber(arguments[kDimArg]) && isPositiveNumber(arguments[kThreadArg])) {
                *real_dimension = atoi(argv[kDimArg]);
                *real_num_threads = atoi(argv[kThreadArg]);
                if (*real_dimension && *real_num_threads) {
                    printOkMessage(*real_dimension, *real_num_threads);
                    return ArgumentsCase::kOk;
                }
            }
        }
    }
    printWrongArguments(argc, argv);
    return ArgumentsCase::kWrongArguments;
}

}  // namespace ArgumentsCheck
