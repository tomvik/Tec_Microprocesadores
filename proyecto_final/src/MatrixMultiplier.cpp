#include <MatrixMultiplier/MatrixMultiplier.h>
#include <ScopeTimer/ScopeTimer.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace MatrixMultiplier {

MatrixMultiplier::MatrixMultiplier() {}

MatrixMultiplier::~MatrixMultiplier() {}

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

void MatrixMultiplier::writeToOutputFile(double** matrix, const std::pair<int, int>& dimension,
                                         const std::string& file_path) {
    std::cout << "Now the result will be written to the output file.\n";
    std::ofstream output_file;

    output_file.open(file_path, std::ios::out | std::ios::trunc);
    output_file << std::fixed;
    output_file << std::setprecision(precision);

    for (int row = 0; row < dimension.first; ++row) {
        for (int col = 0; col < dimension.second; ++col) {
            output_file << matrix[row][col] << "\n";
        }
    }

    output_file.close();
}

bool MatrixMultiplier::compareToOutputFile(double** matrix, const std::pair<int, int>& dimension,
                                           const std::string& file_path) {
    std::cout << "Now the result will be verified.\n";
    std::ifstream result_file;

    result_file.open(file_path);

    if (!result_file.is_open() || result_file.fail()) {
        writeToOutputFile(matrix, dimension, file_path);
        return true;
    }

    double current_value;

    for (int row = 0; row < dimension.first; ++row) {
        for (int col = 0; col < dimension.second; ++col) {
            std::string line;
            if (!std::getline(result_file, line)) {
                std::cerr << "Not enough lines in the file" << std::endl;
                return false;
            }
            std::istringstream iss(line);
            iss >> current_value;

            if (std::abs(matrix[row][col] - current_value) > acceptable_difference) {
                std::cout << "Wrong value " << matrix[row][col] << " " << current_value
                          << std::endl;
                return false;
            }
        }
    }
    return true;
}

void MatrixMultiplier::multiplyNTimes(double** matrix_a, double** matrix_b, double** matrix_c,
                                      const std::vector<std::pair<int, int>>& dimensions,
                                      const int number_of_runs, const std::string& file_path) {
    std::cout << "\n\n*** It will run " << number_of_runs << " times the " << getMethodName()
              << " method with " << getThreadsAmount() << " thread(s)\n";
    for (int run = 0; run < number_of_runs; ++run) {
        ScopeTimer::ScopeTimer timer("multiplyNTimes", true);

        multiply(matrix_a, matrix_b, matrix_c, dimensions);
        run_times_.emplace_back(timer.getDuration());

        std::cout << "For the iteration #" << run + 1 << " it took "
                  << run_times_[run_times_.size() - 1] << " seconds.\n";

        if (!output_file_has_been_opened) {
            writeToOutputFile(matrix_c, dimensions[2], file_path);
            output_file_has_been_opened = true;
        } else {
            if (compareToOutputFile(matrix_c, dimensions[2], file_path)) {
                std::cout << "It gave the same answer up to 10 decimal places\n";
            } else {
                std::cout << "Wrong answer\n";
            }
        }
    }
}

}  // namespace MatrixMultiplier
