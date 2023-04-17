/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "Util.h"
#include <Corrade/Utility/Debug.h>
#include <Magnum/Math/Algorithms/GramSchmidt.h>
#include <stdexcept>

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
RepeatTask::RepeatTask(std::function<void()>&& callback)
: m_callback(std::move(callback)) {}

// -----------------------------------------------------------------------------
RepeatTask::~RepeatTask() {
    stop();
}

// -----------------------------------------------------------------------------
void RepeatTask::start(Timer::duration_t interval) {
    if (m_thread!=nullptr) {
        throw std::runtime_error("RepeatTask::start called on already running task");
    }
    m_interval = interval;
    m_keepRunning = true;
    m_timer.reset();
    m_thread = std::make_unique<std::thread>([this]{run();});
}

// -----------------------------------------------------------------------------
void RepeatTask::stop() {
    m_keepRunning = false;
    if (m_thread==nullptr) {return;}
    m_thread->join();
    m_thread.reset();
}

// -----------------------------------------------------------------------------
void RepeatTask::run() {
    while (m_keepRunning) {
        Timer::duration_t timeLeft{m_interval - m_timer.peek()};
        if (timeLeft < Timer::duration_t{0}) {
            m_timer.reset();
            m_callback();
        } else {
            std::this_thread::sleep_for(timeLeft);
        }
    }
}

// -----------------------------------------------------------------------------
Magnum::Matrix3 outerProduct(const Magnum::Vector3& a, const Magnum::Vector3& b) {
    return Magnum::Matrix3{ // construct from *column* vectors
        {a[0]*b[0], a[1]*b[0], a[2]*b[0]},
        {a[0]*b[1], a[1]*b[1], a[2]*b[1]},
        {a[0]*b[2], a[1]*b[2], a[2]*b[2]}
    };
}

// -----------------------------------------------------------------------------
float round(float x) {
    if (x < 1.0/RoundingPrecision && x > -1.0/RoundingPrecision) {return 0.0f;}
    return std::round(RoundingPrecision*x)/RoundingPrecision;
}

// -----------------------------------------------------------------------------
Magnum::Vector3 round(const Magnum::Vector3& v) {
    return Magnum::Vector3{
        round(v[0]), round(v[1]), round(v[2])
    };
}

// -----------------------------------------------------------------------------
Magnum::Vector4 round(const Magnum::Vector4& v) {
    return Magnum::Vector4{
        round(v[0]), round(v[1]), round(v[2]), round(v[3])
    };
}

// -----------------------------------------------------------------------------
Magnum::Matrix3 round(const Magnum::Matrix3& m) {
    return Magnum::Matrix3{
        round(m[0]), round(m[1]), round(m[2])
    };
}

// -----------------------------------------------------------------------------
Magnum::Matrix4 round(const Magnum::Matrix4& m) {
    return Magnum::Matrix4{
        round(m[0]), round(m[1]), round(m[2]), round(m[3])
    };
}

// -----------------------------------------------------------------------------
void orthonormaliseRotation(Magnum::Matrix4 &trfMatrix) {
    Magnum::Matrix3 rot{
        Magnum::Vector3{trfMatrix[0][0], trfMatrix[0][1], trfMatrix[0][2]}.normalized(),
        Magnum::Vector3{trfMatrix[1][0], trfMatrix[1][1], trfMatrix[1][2]}.normalized(),
        Magnum::Vector3{trfMatrix[2][0], trfMatrix[2][1], trfMatrix[2][2]}.normalized()
    };
    if (rot.isOrthogonal()) {return;}
    Magnum::Math::Algorithms::gramSchmidtOrthonormalizeInPlace(rot);
    trfMatrix[0] = {rot[0][0], rot[0][1], rot[0][2], 0.0f};
    trfMatrix[1] = {rot[1][0], rot[1][1], rot[1][2], 0.0f};
    trfMatrix[2] = {rot[2][0], rot[2][1], rot[2][2], 0.0f};
}

} // namespace CollisionSim::Util
