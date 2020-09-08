#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cctype>
#include <chrono>
#include <future>
#include <vector>

#include "ArgumentsCheck.h"

bool verifyResult(const std::vector<std::vector<int>>& C, const int dimension) {
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            if (C[row][col] != ((row + col) * 2)) {
                return false;
            }
        }
    }
    return true;
}

void createMatrices(std::vector<std::vector<int>>* A, std::vector<std::vector<int>>* B,
                    std::vector<std::vector<int>>* C, const int dimension) {
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            (*A)[row][col] = (row + col);
            (*B)[row][col] = (row + col);
        }
    }
}

void simpleAdd(std::vector<std::vector<int>>* C, const std::vector<std::vector<int>>& A,
               const std::vector<std::vector<int>>& B, const int dimension) {
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            (*C)[row][col] = A[row][col] + B[row][col];
        }
    }
}

int main(int argc, char** argv) {
    int dimension, num_threads;
    ArgumentsCheck::ArgumentsCase argument_case =
        ArgumentsCheck::handleArguments(&dimension, &num_threads, argc, argv);

    switch (argument_case) {
        case ArgumentsCheck::ArgumentsCase::kHelp:
            return 0;
        case ArgumentsCheck::ArgumentsCase::kWrongArguments:
            return 1;
        case ArgumentsCheck::ArgumentsCase::kOk:
        default:
            break;
    }

    std::vector<std::vector<int>> A(dimension, std::vector<int>(dimension, 0));
    std::vector<std::vector<int>> B(dimension, std::vector<int>(dimension, 0));
    std::vector<std::vector<int>> C(dimension, std::vector<int>(dimension, 0));
    time_t start, end;

    createMatrices(&A, &B, &C, dimension);

    start = clock();

    simpleAdd(&C, A, B, dimension);

    end = clock();

    if (verifyResult(C, dimension)) {
        printf("Results verified!!! (%ld)\n", (int64_t)(end - start));
    } else {
        printf("Wrong results!!!\n");
    }

    return 0;
}
