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

// -----------------------------------------------------------------------------
Magnum::Matrix3 outerProduct(const Magnum::Vector3& a, const Magnum::Vector3& b) {
    return Magnum::Matrix3{ // construct from *column* vectors
        {a[0]*b[0], a[1]*b[0], a[2]*b[0]},
        {a[0]*b[1], a[1]*b[1], a[2]*b[1]},
        {a[0]*b[2], a[1]*b[2], a[2]*b[2]}
    };
}

} // namespace CollisionSim::Util
