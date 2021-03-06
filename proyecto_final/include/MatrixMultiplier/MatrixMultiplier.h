
#ifndef PROYECTO_FINAL_INCLUDE_MATRIXMULTIPLIER_MATRIXMULTIPLIER_H_
#define PROYECTO_FINAL_INCLUDE_MATRIXMULTIPLIER_MATRIXMULTIPLIER_H_

#include <string>
#include <utility>
#include <vector>

namespace MatrixMultiplier {

class MatrixMultiplier {
 public:
    // Creates a MatrixMultiplier.
    MatrixMultiplier(const std::string& method_name, const int amount_threads);

    // Returns all the runtimes.
    const std::vector<double> getRunTimes();

    // Returns the average runtime.
    const double getAverageRunTime();

    // Returns the name of the method being used.
    const std::string getMethodName();

    // Returns the number of threads being used.
    const int getThreadsAmount();

    // Will multiply the matrices number_of_runs times;
    void multiplyNTimes(double** matrix_a, double** matrix_b, double** matrix_c,
                        const std::vector<std::pair<int, int>>& dimensions,
                        const int number_of_runs, const std::string& file_path);

    // Destructor of the MatrixMultiplier.
    ~MatrixMultiplier();

 protected:
    // Multiplies matrix_a and matrix_b storing it in matrix_c
    // By this point it is assured that the matrices are correct and can be multiplied.
    virtual void multiply(double** matrix_a, double** matrix_b, double** matrix_c,
                          const std::vector<std::pair<int, int>>& dimensions) = 0;

    // Writes the content of the matrix to the output file specified.
    // It overrides such file if it already exists, otherwise it creates a new one.
    void writeToOutputFile(double** matrix, const std::pair<int, int>& dimension,
                           const std::string& file_path);

    // Compares the given matrix with the resulting matrix in the file_path.
    // It returns true if it was the same and false otherwise.
    // Also if the file does not exist, it creates a new one and returns true.
    // If the file lines does not match the elements of the resulting matrix, it returns false.
    bool compareToOutputFile(double** matrix, const std::pair<int, int>& dimension,
                             const std::string& file_path);

    std::vector<double> run_times_;
    std::string method_name_;
    int amount_of_threads_;

    static bool output_file_has_been_opened;

    const int precision = 10;
    const double acceptable_difference = 0.0000000001;  // 1*10^-10
};

}  // namespace MatrixMultiplier

#endif  // PROYECTO_FINAL_INCLUDE_MATRIXMULTIPLIER_MATRIXMULTIPLIER_H_
