
#ifndef PROYECTO_FINAL_INCLUDE_MATRIXMULTIPLIER_MATRIXMULTIPLIER_H_
#define PROYECTO_FINAL_INCLUDE_MATRIXMULTIPLIER_MATRIXMULTIPLIER_H_

#include <string>
#include <utility>
#include <vector>

namespace MatrixMultiplier {

class MatrixMultiplier {
 public:
    // Creates a MatrixMultiplier.
    MatrixMultiplier();

    // Returns all the runtimes.
    const std::vector<double> getRunTimes();

    // Returns the average runtime.
    const double getAverageRunTime();

    // Will multiply the matrices number_of_runs times;
    void multiplyNTimes(double** matrix_a, double** matrix_b, double** matrix_c,
                        const std::vector<std::pair<int, int>>& dimensions,
                        const int number_of_runs);

    // Destructor of the MatrixMultiplier.
    ~MatrixMultiplier();

 protected:
    // Multiplies matrix_a and matrix_b storing it in matrix_c
    // By this point it is assured that the matrices are correct and can be multiplied.
    virtual void multiply(double** matrix_a, double** matrix_b, double** matrix_c,
                          const std::vector<std::pair<int, int>>& dimensions) = 0;

    std::vector<double> run_times_;
};

}  // namespace MatrixMultiplier

#endif  // PROYECTO_FINAL_INCLUDE_MATRIXMULTIPLIER_MATRIXMULTIPLIER_H_
