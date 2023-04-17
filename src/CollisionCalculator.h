/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_COLLISIONCALCULATOR
#define COLLISION_SIM_COLLISIONCALCULATOR

#include "Actor.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Range.h>
#include <vector>

namespace CollisionSim::CollisionCalculator {

void collideWorldSequential(std::vector<Actor>& actors, const Magnum::Range3D& worldBoundaries);
void collideWorldParallel(std::vector<Actor>& actors, const Magnum::Range3D& worldBoundaries, size_t numAllVertices);

} // namespace CollisionSim::CollisionCalculator

#endif // COLLISION_SIM_COLLISIONCALCULATOR
