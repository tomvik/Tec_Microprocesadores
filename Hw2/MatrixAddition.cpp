#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cctype>
#include <chrono>  // NOLINT [build/c++11]
#include <future>
#include <vector>

#include "ArgumentsCheck.h"
#include "Timer.h"

void printMatrix(const std::string s, const std::vector<std::vector<int>>& m) {
    std::cout << "printing matrix " << s << ":\n";
    for (const auto row : m) {
        for (const auto curr : row) {
            std::cout << curr << " ";
        }
        std::cout << "\n";
    }
}

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
    Timer::Timer timer("simpleAdd");
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            (*C)[row][col] = A[row][col] + B[row][col];
        }
    }
}

void addRows(std::vector<std::vector<int>>::iterator c_it_in,
             std::vector<std::vector<int>>::const_iterator a_it_in,
             std::vector<std::vector<int>>::const_iterator b_it_in, int initial, int limit) {
    for (int row = initial; row < limit; ++row) {
        for (int col = 0; col < c_it_in->size(); ++col) {
            (*c_it_in)[col] = (*a_it_in)[col] + (*b_it_in)[col];
        }
        ++c_it_in;
        ++a_it_in;
        ++b_it_in;
    }
}

void multiAdd(std::vector<std::vector<int>>* C, const std::vector<std::vector<int>>& A,
              const std::vector<std::vector<int>>& B, const int dimension, const int num_threads) {
    Timer::Timer timer("multiAdd");

    std::vector<std::thread> threads;

    const int step = dimension / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(std::thread(addRows, (C->begin() + (i * step)),
                                         (A.begin() + (i * step)), (B.begin() + (i * step)),
                                         i * step, (i + 1) * step));
    }

    for (auto& thread : threads) {
        thread.join();
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

    createMatrices(&A, &B, &C, dimension);

    // simpleAdd(&C, A, B, dimension);
    multiAdd(&C, A, B, dimension, num_threads);

    if (verifyResult(C, dimension)) {
        printf("Results verified!!!\n");
    } else {
        printf("Wrong results!!!\n");
    }

    return 0;
}
