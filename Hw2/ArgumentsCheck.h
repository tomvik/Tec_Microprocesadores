#ifndef HW2_ARGUMENTSCHECK_H_
#define HW2_ARGUMENTSCHECK_H_

#include <algorithm>
#include <iostream>
#include <string>

namespace ArgumentsCheck {

// Enum class for the possible argument cases.
enum class ArgumentsCase { kOk = 0, kHelp = 1, kWrongArguments = 2 };

// Checks if the arguments are correct, and prints the appropiate message if neccesary.
// It also updates the value of the variables real_dimension and real_num_threads.
ArgumentsCase handleArguments(int* real_dimension, int* real_num_threads, int argc,
                              char** argv);

}  // namespace ArgumentsCheck

#endif  // HW2_ARGUMENTSCHECK_H_
