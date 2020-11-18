#include <MatrixMultiplier/PThreadMultiplier.h>
#include <omp.h>
#include <pthread.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {
struct pThreadStruct {
    double** matrix_a;
    double** matrix_b;
    double** matrix_c;

    int row_start;
    int row_end;

    int matrix_a_cols;
    int matrix_b_cols;
};

void* singleThreadMul(void* arg) {
    pThreadStruct thread_data = *reinterpret_cast<pThreadStruct*>(arg);

    for (int row_a = thread_data.row_start; row_a < thread_data.row_end; ++row_a) {
        for (int col_b = 0; col_b < thread_data.matrix_b_cols; ++col_b) {
            double acum = 0;
            for (int col_a = 0; col_a < thread_data.matrix_a_cols; ++col_a) {
                acum += thread_data.matrix_a[row_a][col_a] * thread_data.matrix_b[col_a][col_b];
            }
            thread_data.matrix_c[row_a][col_b] = acum;
        }
    }
}

}  // namespace

namespace MatrixMultiplier {

PThreadMultiplier::PThreadMultiplier(const int amount_of_threads)
    : MatrixMultiplier("PThread", amount_of_threads) {}

void PThreadMultiplier::multiply(double** matrix_a, double** matrix_b, double** matrix_c,
                                 const std::vector<std::pair<int, int>>& dimensions) {
    const int matrix_a_rows = dimensions[0].first;
    const int matrix_a_cols = dimensions[0].second;
    const int matrix_b_cols = dimensions[1].second;

    const int step = matrix_a_rows / amount_of_threads_;

    const int kThreads = amount_of_threads_;

    pThreadStruct params[kThreads];
    pthread_t pthreadID[kThreads];

    for (int i = 0; i < kThreads; ++i) {
        params[i].matrix_a = matrix_a;
        params[i].matrix_b = matrix_b;
        params[i].matrix_c = matrix_c;

        params[i].matrix_a_cols = matrix_a_cols;
        params[i].matrix_b_cols = matrix_b_cols;

        params[i].row_start = i * step;
        params[i].row_end = i == kThreads - 1 ? matrix_a_rows : (i + 1) * step;
        pthread_create(&(pthreadID[i]), NULL, singleThreadMul, &(params[i]));
    }

    for (int i = 0; i < kThreads; ++i) {
        pthread_join(pthreadID[i], NULL);
    }
}

}  // namespace MatrixMultiplier
