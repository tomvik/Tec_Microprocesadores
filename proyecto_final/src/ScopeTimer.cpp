#include <ScopeTimer/ScopeTimer.h>

#include <chrono>

#include <stdio.h>
#include <iostream>

namespace ScopeTimer {

ScopeTimer::ScopeTimer(const std::string& name) : name_(name) {
    initial_time_ = std::chrono::high_resolution_clock::now();
}

ScopeTimer::~ScopeTimer() {
    std::cout << "The total execution time for " << name_ << " was: " << getDuration() << "s\n";
}

const double ScopeTimer::getDuration() {
    const std::chrono::duration<double> duration =
        (std::chrono::high_resolution_clock::now() - initial_time_);

    return duration.count();
}

}  // namespace ScopeTimer
