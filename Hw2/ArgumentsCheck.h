#ifndef HW2_ARGUMENTSCHECK_H_
#define HW2_ARGUMENTSCHECK_H_

#include <algorithm>
#include <iostream>
#include <string>

namespace ArgumentsCheck {

enum class ArgumentsCase { kOk = 0, kHelp = 1, kWrongArguments = 2 };

ArgumentsCase handleArguments(int* real_dimension, int* real_num_threads, int argc,
                              char** argv);

}  // namespace ArgumentsCheck

#endif  // HW2_ARGUMENTSCHECK_H_
