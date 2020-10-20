
#ifndef HW6_INCLUDE_TIMER_TIMER_H_
#define HW6_INCLUDE_TIMER_TIMER_H_

#include <chrono>  // NOLINT [build/c++11]
#include <string>

namespace Timer {

class Timer {
 public:
    // Creates a timer with the name given.
    explicit Timer(const std::string& name);

    // Destructor of the timer, which will also print out the amount of time it went
    // since its construction.
    ~Timer();

 private:
    // Returns the duration since the initial time in seconds.
    const double getDuration();

    std::string name_;
    std::chrono::_V2::system_clock::time_point initial_time_;
};

}  // namespace Timer

#endif  // HW6_INCLUDE_TIMER_TIMER_H_
