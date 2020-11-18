#include <ArgumentsCheck/ArgumentsCheck.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void printOkMessage(const std::string& output_file_path) {
    std::cout << "The two files were opened successfully\n"
              << "And the output file will be written in: " << output_file_path << "\n";
}

void printHelpMessage(char** argv) {
    std::cerr << "Usages: " << argv[0] << " {<option>} or {<MatrixA> <MatrixB>}\n"
              << "Option:\n"
              << "\t-h,--help\t\tShow this help message\n"
              << "MatrixA:\n"
              << "\tString relative or absolute path to the MatrixA.txt\n"
              << "\tNOTE: this path will also be used for the matrixC.txt\n"
              << "MatrixB:\n"
              << "\tString relative or absolute path to the MatrixB.txt\n"
              << std::endl;
}

void printWrongArguments(int argc, char** argv) {
    std::cerr << "[ ERROR ] "
              << "The following execution is incorrect: ";
    for (int i = 0; i < argc; ++i) {
        std::cerr << argv[i] << " ";
    }
    std::cerr << "\n"
              << "          "
              << "Run the help option to check the correct execution:\n"
              << "          "
              << "\t" << argv[0] << " -h" << std::endl;
}

void printWrongPathOrFile(const std::string& path) {
    std::cerr << "[ ERROR ] "
              << "Something failed opening the input file " << path << "\n"
              << "          "
              << "Check that the file exists and that the path is correct.\n";
}

}  // namespace

namespace ArgumentsCheck {

ArgumentsCase handleArgumentsAndGetFileHandles(const int argc, char** argv,
                                               std::vector<std::ifstream>* input_files,
                                               std::string* output_file_path) {
    std::vector<std::string> arguments;
    for (int i = 1; i < argc; ++i) {
        arguments.push_back(argv[i]);
    }

    if (arguments.size() == 2) {
        for (int i = 0; i < arguments.size(); ++i) {
            (*input_files)[i].open(arguments[i], std::ios::in | std::ios::_Nocreate);
            if ((*input_files)[i].fail()) {
                printWrongPathOrFile(arguments[i]);
                return ArgumentsCase::kWrongPathOrFile;
            }
            std::cout << "Finished opening the input file " << arguments[i] << std::endl;
        }
        const size_t pos_relative_path = arguments[0].find_last_of('/') + 1;
        *output_file_path = arguments[0].substr(0, pos_relative_path) + "matrixC.txt";
        printOkMessage(*output_file_path);
        return ArgumentsCase::kOk;
    } else if (arguments.size() == 1 && (arguments[0] == "-h" || arguments[0] == "--help")) {
        printHelpMessage(argv);
        return ArgumentsCase::kHelp;
    }
    printWrongArguments(argc, argv);
    return ArgumentsCase::kWrongArguments;
}

}  // namespace ArgumentsCheck
