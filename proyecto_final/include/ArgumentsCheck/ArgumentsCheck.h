#ifndef PROYECTO_FINAL_INCLUDE_ARGUMENTSCHECK_ARGUMENTSCHECK_H_
#define PROYECTO_FINAL_INCLUDE_ARGUMENTSCHECK_ARGUMENTSCHECK_H_

#include <fstream>
#include <vector>
namespace ArgumentsCheck {

// Enum class for the possible argument cases.
enum class ArgumentsCase { kOk = 0, kHelp = 1, kWrongArguments = 2, kWrongPathOrFile = 3 };

// Checks if the arguments are correct, and prints the appropiate message if neccesary.
// It also updates the values of the files handles.
ArgumentsCase handleArgumentsAndGetFileHandles(const int argc, char** argv,
                                               std::vector<std::ifstream>* input_files);

}  // namespace ArgumentsCheck

#endif  // PROYECTO_FINAL_INCLUDE_ARGUMENTSCHECK_ARGUMENTSCHECK_H_
