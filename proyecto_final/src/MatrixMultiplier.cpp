#include <MatrixMultiplier/MatrixMultiplier.h>
#include <ScopeTimer/ScopeTimer.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace MatrixMultiplier {

MatrixMultiplier::MatrixMultiplier() {}

bool MatrixMultiplier::output_file_has_been_opened = false;

const std::vector<double> MatrixMultiplier::getRunTimes() { return run_times_; }

const double MatrixMultiplier::getAverageRunTime() {
    if (run_times_.size() == 0) {
        return 0;
    }

    double sum = 0;
    for (const auto element : run_times_) {
        sum += element;
    }
    return sum / run_times_.size();
}

void MatrixMultiplier::multiplyNTimes(double** matrix_a, double** matrix_b, double** matrix_c,
                                      const std::vector<std::pair<int, int>>& dimensions,
                                      const int number_of_runs) {
    for (int run = 0; run < number_of_runs; ++run) {
        ScopeTimer::ScopeTimer timer("multiplyNTimes");
        multiply(matrix_a, matrix_b, matrix_c, dimensions);
        run_times_.emplace_back(timer.getDuration());

        if (!output_file_has_been_opened) {
            std::cout << "First\n";
            output_file_has_been_opened = true;
        } else {
            std::cout << "Not first\n";
        }
    }
}

}  // namespace MatrixMultiplier
