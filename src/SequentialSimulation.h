/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_SEQUENTIALSIMULATION
#define COLLISION_SIM_SEQUENTIALSIMULATION

#include "SequentialState.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Range.h>
#include <vector>

namespace CollisionSim {
class Actor;
}

namespace CollisionSim::Simulation {

namespace detail {
void simulateMotionSequential(float dtime, std::vector<Actor>& actors, const Magnum::Range3D& worldBoundaries);
void collideWorldSequential(std::vector<Actor>& actors, const Magnum::Range3D& worldBoundaries);
void collideBroadSequential(std::vector<Actor>& actors, SequentialState& state);
void collideNarrowSequential(std::vector<Actor>& actors, SequentialState& state);
void impulseCollision(Actor& a, Actor& b, const Magnum::Vector3& point, const Magnum::Vector3& normal);
}

void simulateSequential(float dtime, std::vector<Actor>& actors, SequentialState& state);

} // namespace CollisionSim::Simulation

#endif // COLLISION_SIM_SEQUENTIALSIMULATION
