/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_CONSTANTS
#define COLLISION_SIM_CONSTANTS

#include "Util.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <chrono>
#include <string_view>

namespace CollisionSim::Constants {

/// Application name shown in the window title bar
constexpr static std::string_view ApplicationName{"Collision Simulation"};

/// Minimum amount of time between two calls to compute state of the simulation
constexpr static Util::Timer::duration_t ComputeInterval{std::chrono::microseconds{0}};

/// Minimum amount of time between two updates of the on-screen text (e.g. FPS counter)
constexpr static Util::Timer::duration_t TextUpdateInterval{std::chrono::milliseconds{200}};

/// Number of measurements to keep for averaging the FPS measurement
constexpr static size_t FrameTimeCounterWindow{2048};

/// Slow down or speed up the simulation with respect to real time
constexpr static float RealTimeScale{0.5f};

/// Scaling factors with respect to SI units
namespace Units {
constexpr static float Distance{100.f}; // cm
constexpr static float Area{Distance*Distance}; // cm^2
constexpr static float Volume{Distance*Distance*Distance}; // cm^3
constexpr static float Mass{1.0f}; // kg
constexpr static float Time{1.0f}; // s
constexpr static float Density{Mass/Volume}; // kg/cm^3
constexpr static float Velocity{Distance/Time}; // cm/s
}

/// Default dimensions of the world
constexpr static Magnum::Vector3 DefaultWorldDimensions{10.0f,8.0f,10.0f};

/// Uniform density of the body materials
constexpr static float DefaultDensity{1000.0f * Units::Density}; // kg/cm^3, approx. water density

/// Gravity
constexpr static float EarthGravity{-9.81f * Units::Distance / (Units::Time * Units::Time)}; // m/s^2

/// Restitution coefficient (fraction of kinematic energy conserved in a collision)
constexpr static float RestitutionCoefficient{0.95f};

} // namespace CollisionSim::Constants


#endif // COLLISION_SIM_CONSTANTS
