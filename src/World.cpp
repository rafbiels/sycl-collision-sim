/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "World.h"

// -----------------------------------------------------------------------------
CollisionSim::World::World(float windowAspectRatio, float gravity)
: m_gravity{gravity} {
    using namespace Magnum::Math::Literals;
    m_projection = Magnum::Matrix4::perspectiveProjection(90.0_degf, windowAspectRatio, 0.01, 100.0)
        * Magnum::Matrix4::translation(Magnum::Vector3::zAxis(-10.0));
}

// -----------------------------------------------------------------------------
Magnum::Matrix4& CollisionSim::World::projection() {
    return m_projection;
}

// -----------------------------------------------------------------------------
void CollisionSim::World::projection(Magnum::Matrix4& proj) {
    m_projection = proj;
}
