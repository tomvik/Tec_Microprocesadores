
#ifndef HW2_TIMER_H_
#define HW2_TIMER_H_

#include <chrono>  // NOLINT [build/c++11]
#include <string>

namespace Timer {

class Timer {
 public:
    explicit Timer(const std::string& name);
    ~Timer();

 private:
    const double getDuration();

    std::string name_;
    std::chrono::_V2::system_clock::time_point initial_time_;
};
    
} // namespace Timer

#endif  // HW2_TIMER_H_