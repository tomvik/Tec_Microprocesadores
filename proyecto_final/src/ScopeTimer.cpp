#include <ScopeTimer/ScopeTimer.h>
#include <stdio.h>

#include <chrono>  // NOLINT [build/c++11]
#include <iostream>

namespace ScopeTimer {

ScopeTimer::ScopeTimer(const std::string& name, const bool silent) : name_(name), silent_(silent) {
    initial_time_ = std::chrono::high_resolution_clock::now();
}

ScopeTimer::~ScopeTimer() {
    if (!silent_) {
        std::cout << "The total execution time for " << name_ << " was: " << getDuration() << "s\n";
    }
}

const double ScopeTimer::getDuration() {
    const std::chrono::duration<double> duration =
        (std::chrono::high_resolution_clock::now() - initial_time_);

    return duration.count();
}

}  // namespace ScopeTimer
