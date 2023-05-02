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
void collideWorldSequential(std::vector<Actor>& actors, const Magnum::Range3D& worldBoundaries);
void collideBroadSequential(std::vector<Actor>& actors, SequentialState* state);

void simulateSequential(float dtime, std::vector<Actor>& actors, SequentialState* state);
void simulateParallel(float dtime, std::vector<Actor>& actors, ParallelState* state, sycl::queue* queue);

} // namespace CollisionSim::Simulation

#endif // COLLISION_SIM_SIMULATION
