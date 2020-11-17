
#ifndef PROYECTO_FINAL_INCLUDE_SCOPETIMER_SCOPETIMER_H_
#define PROYECTO_FINAL_INCLUDE_SCOPETIMER_SCOPETIMER_H_

#include <chrono>
#include <string>

namespace ScopeTimer {
    
class ScopeTimer {
 public:
    // Creates a ScopeTimer with the name given.
    explicit ScopeTimer(const std::string& name);

    // Destructor of the ScopeTimer, which will also print out the amount of time it went
    // since its construction.
    ~ScopeTimer();

 private:
    // Returns the duration since the initial time in seconds.
    const double getDuration();

    std::string name_;
    std::chrono::steady_clock::time_point initial_time_;
};

}  // namespace ScopeTimer

#endif  // PROYECTO_FINAL_INCLUDE_SCOPETIMER_SCOPETIMER_H_
