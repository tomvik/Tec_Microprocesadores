#include <MatrixMultiplier/MatrixMultiplier.h>

#include <ScopeTimer/ScopeTimer.h>

#include <string>
#include <vector>

namespace MatrixMultiplier {

MatrixMultiplier::MatrixMultiplier() {}

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

void MatrixMultiplier::multiplyNTimes(const std::vector<std::vector<double>>& matrix_a,
                    const std::vector<std::vector<double>>& matrix_b,
                    std::vector<std::vector<double>>* matrix_c, const int number_of_runs) {
    for (int run = 0; run < number_of_runs; ++run) {
        ScopeTimer::ScopeTimer timer("multiplyNTimes");
        multiply(matrix_a, matrix_b, matrix_c);
        run_times_.emplace_back(timer.getDuration());
    }
}

}  // namespace MatrixMultiplier
