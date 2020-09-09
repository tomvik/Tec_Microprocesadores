#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cctype>
#include <chrono>  // NOLINT [build/c++11]
#include <future>
#include <vector>

#include "ArgumentsCheck.h"
#include "Timer.h"

int* A = nullptr;
int* B = nullptr;
int* C = nullptr;

void printMatrix(const std::string s, const std::vector<std::vector<int>>& m) {
    std::cout << "printing matrix " << s << ":\n";
    for (const auto row : m) {
        for (const auto curr : row) {
            std::cout << curr << " ";
        }
        std::cout << "\n";
    }
}

bool verifyResultVector(const std::vector<std::vector<int>>& C, const int dimension) {
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            if (C[row][col] != ((row + col) * 2)) {
                return false;
            }
        }
    }
    return true;
}

bool verifyResultPointer(int* C, const int dimension) {
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            if (*(C + row * dimension + col) != ((row + col) * 2)) {
                return false;
            }
        }
    }
    return true;
}

void createMatricesVector(std::vector<std::vector<int>>* A, std::vector<std::vector<int>>* B,
                          std::vector<std::vector<int>>* C, const int dimension) {
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            (*A)[row][col] = (*B)[row][col] = (row + col);
        }
    }
}

void createMatricesPointer(int* A, int* B, int* C, const int dimension) {
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            *(A + row * dimension + col) = *(B + row * dimension + col) = row + col;
            *(C + row * dimension + col) = 0;
        }
    }
}

void simpleAddVector(std::vector<std::vector<int>>* C, const std::vector<std::vector<int>>& A,
                     const std::vector<std::vector<int>>& B, const int dimension) {
    Timer::Timer timer("simpleAddVector");
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            (*C)[row][col] = A[row][col] + B[row][col];
        }
    }
}

void simpleAddPointer(int* C, int* A, int* B, const int dimension) {
    Timer::Timer timer("simpleAddPointer");
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            *(C + row * dimension + col) =
                *(A + row * dimension + col) + *(B + row * dimension + col);
        }
    }
}

void addRowsVector(std::vector<std::vector<int>>::iterator c_it_in,
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

void multiAddVector(std::vector<std::vector<int>>* C, const std::vector<std::vector<int>>& A,
                    const std::vector<std::vector<int>>& B, const int dimension,
                    const int num_threads) {
    Timer::Timer timer("multiAddVector");

    std::vector<std::thread> threads;

    const int step = dimension / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(std::thread(addRowsVector, (C->begin() + (i * step)),
                                         (A.begin() + (i * step)), (B.begin() + (i * step)),
                                         i * step, (i + 1) * step));
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

void addRowsPointer(int initial, int limit, int dimension) {
    for (int row = initial; row < limit; ++row) {
        for (int col = 0; col < dimension; ++col) {
            *(C + row * dimension + col) =
                *(A + row * dimension + col) + *(B + row * dimension + col);
        }
    }
}

void multiAddPointer(const int dimension, const int num_threads) {
    Timer::Timer timer("multiAddPointer");

    std::vector<std::thread> threads;

    const int step = dimension / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(std::thread(addRowsPointer, i * step, (i + 1) * step, dimension));
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

void threadAddVector(const int dimension, const int num_threads) {
    std::vector<std::vector<int>> A(dimension, std::vector<int>(dimension, 0));
    std::vector<std::vector<int>> B(dimension, std::vector<int>(dimension, 0));
    std::vector<std::vector<int>> C(dimension, std::vector<int>(dimension, 0));

    createMatricesVector(&A, &B, &C, dimension);

    // simpleAddVector(&C, A, B, dimension);
    multiAddVector(&C, A, B, dimension, num_threads);

    if (verifyResultVector(C, dimension)) {
        printf("Results verified!!!\n");
    } else {
        printf("Wrong results!!!\n");
    }
}

void threadAddPointer(const int dimension, const int num_threads) {
    size_t datasize = sizeof(int) * dimension * dimension;

    A = (int*)malloc(datasize);
    B = (int*)malloc(datasize);
    C = (int*)malloc(datasize);

    createMatricesPointer(A, B, C, dimension);

    // simpleAddPointer(C, A, B, dimension);
    multiAddPointer(dimension, num_threads);

    if (verifyResultPointer(C, dimension)) {
        printf("Results verified!!!\n");
    } else {
        printf("Wrong results!!!\n");
    }

    free(A);
    free(B);
    free(C);
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

    // threadAddVector(dimension, num_threads);
    threadAddPointer(dimension, num_threads);

    return 0;
}
