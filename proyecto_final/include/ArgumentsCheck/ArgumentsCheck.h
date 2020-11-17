#ifndef PROYECTO_FINAL_INCLUDE_ARGUMENTSCHECK_ARGUMENTSCHECK_H_
#define PROYECTO_FINAL_INCLUDE_ARGUMENTSCHECK_ARGUMENTSCHECK_H_

namespace ArgumentsCheck {

// Enum class for the possible argument cases.
enum class ArgumentsCase { kOk = 0, kHelp = 1, kWrongArguments = 2 };

// Checks if the arguments are correct, and prints the appropiate message if neccesary.
// It also updates the value of the variables real_dimension and real_num_threads.
ArgumentsCase handleArguments(int argc, char** argv);

}  // namespace ArgumentsCheck

#endif  // PROYECTO_FINAL_INCLUDE_ARGUMENTSCHECK_ARGUMENTSCHECK_H_
