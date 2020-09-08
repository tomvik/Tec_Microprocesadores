#include "Timer.h"

#include <chrono>  // NOLINT [build/c++11]
#include <iostream>
#include <string>

namespace Timer {

Timer::Timer(const std::string& name) : name_(name) {
    initial_time_ = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() {
    std::cout << "The total execution time for " << name_ << " was: " << getDuration() << "s\n";
}

const double Timer::getDuration() {
    const std::chrono::duration<double> duration =
        (std::chrono::high_resolution_clock::now() - initial_time_);

    return duration.count();
}

}  // namespace Timer