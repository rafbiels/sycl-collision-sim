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

} // namespace CollisionSim::Constants


#endif // COLLISION_SIM_CONSTANTS
