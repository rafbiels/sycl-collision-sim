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

constexpr static std::string_view ApplicationName{"Collision Simulation"};
constexpr static Util::Timer::duration_t TextUpdateInterval{std::chrono::milliseconds{200}};
constexpr static size_t FrameTimeCounterWindow{200};

namespace Units {
constexpr static float Distance{100.f}; // cm
constexpr static float Area{Distance*Distance}; // cm^2
constexpr static float Volume{Distance*Distance*Distance}; // cm^3
constexpr static float Mass{1.0}; // kg
constexpr static float Time{1.0}; // s
constexpr static float Density{Mass/Volume}; // kg/cm^3
}

constexpr static float DefaultDensity{1000.0 * Units::Density}; // kg/cm^3, approx. water density

} // namespace CollisionSim::Constants


#endif // COLLISION_SIM_CONSTANTS
