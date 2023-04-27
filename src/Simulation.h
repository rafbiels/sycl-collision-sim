/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_SIMULATION
#define COLLISION_SIM_SIMULATION

#include "Actor.h"
#include "State.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Range.h>
#include <sycl/sycl.hpp>
#include <vector>

namespace CollisionSim::Simulation {

void simulateMotionSequential(float dtime, std::vector<Actor>& actors);
void simulateMotionParallel(float dtime, sycl::queue* queue, std::vector<Actor>& actors, State* state);
void simulateSequential(float dtime, std::vector<Actor>& actors, const Magnum::Range3D& worldBoundaries);

void simulateParallel(float dtime, sycl::queue* queue, std::vector<Actor>& actors, State* state);

} // namespace CollisionSim::Simulation

#endif // COLLISION_SIM_SIMULATION
