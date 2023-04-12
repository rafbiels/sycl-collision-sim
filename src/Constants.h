/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_CONSTANTS
#define COLLISION_SIM_CONSTANTS

#include "Util.h"
#include <string_view>

namespace CollisionSim::Constants {

/// Application name shown in the window title bar
constexpr static std::string_view ApplicationName{"Collision Simulation"};

/// Minimum amount of time between two updates of the on-screen text (e.g. FPS counter)
constexpr static Util::Timer::duration_t TextUpdateInterval{std::chrono::milliseconds{200}};

/// Number of measurements to keep for averaging the FPS measurement
constexpr static size_t FrameTimeCounterWindow{200};

/// Slow down or speed up the simulation with respect to real time
constexpr static float RealTimeScale{0.05};

/// Scaling factors with respect to SI units
namespace Units {
constexpr static float Distance{100.f}; // cm
constexpr static float Area{Distance*Distance}; // cm^2
constexpr static float Volume{Distance*Distance*Distance}; // cm^3
constexpr static float Mass{1.0}; // kg
constexpr static float Time{1.0}; // s
constexpr static float Density{Mass/Volume}; // kg/cm^3
}

/// Uniform density of the body materials
constexpr static float DefaultDensity{1000.0 * Units::Density}; // kg/cm^3, approx. water density

/// Gravity
constexpr static float EarthGravity{9.81 * Units::Distance / (Units::Time * Units::Time)}; // m/s^2

} // namespace CollisionSim::Constants


#endif // COLLISION_SIM_CONSTANTS
