/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_PARALLELSIMULATION
#define COLLISION_SIM_PARALLELSIMULATION

#include "ParallelState.h"
#include <sycl/sycl.hpp>
#include <vector>

namespace CollisionSim {
class Actor;
}

namespace CollisionSim::Simulation {

void simulateParallel(float dtime, std::vector<Actor>& actors, ParallelState& state, sycl::queue& queue);

} // namespace CollisionSim::Simulation

#endif // COLLISION_SIM_PARALLELSIMULATION
