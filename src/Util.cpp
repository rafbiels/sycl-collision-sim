/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "Util.h"

namespace CollisionSim::Util {

// -----------------------------------------------------------------------------
void Timer::reset() {
    m_currentTime = clock_t::now();
}

// -----------------------------------------------------------------------------
Timer::duration_t Timer::step() {
    time_point_t previousTime = m_currentTime;
    m_currentTime = clock_t::now();
    return m_currentTime - previousTime;
}

// -----------------------------------------------------------------------------
Timer::duration_t Timer::peek() const {
    return clock_t::now() - m_currentTime;
}

// -----------------------------------------------------------------------------
bool Timer::stepIfElapsed(Timer::duration_t duration) {
    if (peek() < duration) {return false;}
    reset();
    return true;
}

} // namespace CollisionSim::Util
