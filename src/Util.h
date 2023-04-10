/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_UTIL
#define COLLISION_SIM_UTIL

#include <chrono>
#include <deque>
#include <numeric>

namespace CollisionSim::Util {

class Timer {
    public:
        using clock_t = std::chrono::steady_clock;
        using time_point_t = clock_t::time_point;
        using duration_t = clock_t::duration;
        /**
         * Reset the stored time to now()
         */
        void reset();
        /**
         * Return the difference from now() to the previously stored
         * time without resetting the stored time
         */
        duration_t peek() const;
        /**
         * Reset the stored time to now() and return the difference
         * to the previously stored time
         */ 
        duration_t step();
        /**
         * Reset the stored time to now() if at least \c duration has
         * elapsed since the previously stored time
         */ 
        bool stepIfElapsed(duration_t duration);
    private:
        time_point_t m_currentTime;
};

template<typename T>
class MovingAverage {
    public:
        MovingAverage(size_t window) : m_window(window) {}
        T value() const {
            return std::accumulate(m_values.begin(), m_values.end(), T{0}) / static_cast<T>(m_values.size());
        }
        void add(T value) {
            if (m_values.size() >= m_window) {
                m_values.pop_front();
            }
            m_values.push_back(value);
        }
        void reset() {m_values.clear();};
    private:
        std::deque<T> m_values;
        size_t m_window{0};
};

} // namespace CollisionSim::Util

#endif // COLLISION_SIM_UTIL
