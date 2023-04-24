/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "State.h"
#include "Util.h"
#include <algorithm>

// -----------------------------------------------------------------------------
CollisionSim::WallCollisionCache::WallCollisionCache(const Magnum::Range3D& worldBoundaries, size_t numAllVertices)
: m_boundaries{worldBoundaries.min()[0], worldBoundaries.max()[0],
               worldBoundaries.min()[1], worldBoundaries.max()[1],
               worldBoundaries.min()[2], worldBoundaries.max()[2]},
  m_collisions(numAllVertices, Wall::None),
  m_addLinearVelocity(numAllVertices, sycl::float3{0.0f}),
  m_addAngularVelocity(numAllVertices, sycl::float3{0.0f}),
  m_boundariesBuf{m_boundaries.data(), m_boundaries.size()},
  m_collisionsBuf{m_collisions.data(), numAllVertices},
  m_addLinearVelocityBuf{m_addLinearVelocity.data(), numAllVertices},
  m_addAngularVelocityBuf{m_addAngularVelocity.data(), numAllVertices} {}

// -----------------------------------------------------------------------------
void CollisionSim::WallCollisionCache::resetBuffers() {
  m_boundariesBuf = {m_boundaries.data(), m_boundaries.size()};
  m_collisionsBuf = {m_collisions.data(), m_collisions.size()};
  m_addLinearVelocityBuf = {m_addLinearVelocity.data(), m_addLinearVelocity.size()};
  m_addAngularVelocityBuf = {m_addAngularVelocity.data(), m_addAngularVelocity.size()};
}

// -----------------------------------------------------------------------------
CollisionSim::State::State(const Magnum::Range3D& worldBoundaries,
                           const std::vector<Actor>& actors,
                           size_t numAllVertices)
: m_numActors{actors.size()},
  m_numAllVertices{numAllVertices},
  m_actorIndices(numAllVertices, uint16_t{0}),
  m_mass(actors.size(), 0.0f),
  m_bodyInertiaInv(actors.size(), {sycl::float3{0.0f},sycl::float3{0.0f},sycl::float3{0.0f}}),
  m_allVertices{std::vector<float>(numAllVertices, 0.0f), std::vector<float>(numAllVertices, 0.0f), std::vector<float>(numAllVertices, 0.0f)},
  m_translation(actors.size(), sycl::float3{0.0f}),
  m_rotation(actors.size(), {sycl::float3{0.0f},sycl::float3{0.0f},sycl::float3{0.0f}}),
  m_inertiaInv(actors.size(), {sycl::float3{0.0f},sycl::float3{0.0f},sycl::float3{0.0f}}),
  m_linearVelocity(actors.size(), sycl::float3{0.0f}),
  m_angularVelocity(actors.size(), sycl::float3{0.0f}),
  m_linearMomentum(actors.size(), sycl::float3{0.0f}),
  m_angularMomentum(actors.size(), sycl::float3{0.0f}),
  m_force(actors.size(), sycl::float3{0.0f}),
  m_torque(actors.size(), sycl::float3{0.0f}),
  m_wallCollisionCache{worldBoundaries, numAllVertices},
  m_actorIndicesBuf{m_actorIndices.data(), numAllVertices},
  m_massBuf{m_mass.data(), actors.size()},
  m_bodyInertiaInvBuf{m_bodyInertiaInv.data(), actors.size()},
  m_vxBuf{m_allVertices[0].data(), numAllVertices},
  m_vyBuf{m_allVertices[1].data(), numAllVertices},
  m_vzBuf{m_allVertices[2].data(), numAllVertices},
  m_translationBuf{m_translation.data(), actors.size()},
  m_rotationBuf{m_rotation.data(), actors.size()},
  m_inertiaInvBuf{m_inertiaInv.data(), actors.size()},
  m_linearVelocityBuf{m_linearVelocity.data(), actors.size()},
  m_angularVelocityBuf{m_angularVelocity.data(), actors.size()},
  m_linearMomentumBuf{m_linearMomentum.data(), actors.size()},
  m_angularMomentumBuf{m_angularMomentum.data(), actors.size()},
  m_forceBuf{m_force.data(), actors.size()},
  m_torqueBuf{m_torque.data(), actors.size()} {

    size_t vertexOffset{0};
    for (size_t iActor{0}; iActor<m_numActors; ++iActor) {
        m_mass[iActor] = actors[iActor].mass();
        m_bodyInertiaInv[iActor] = Util::toSycl(actors[iActor].bodyInertiaInv());
        m_translation[iActor] = Util::toSycl(actors[iActor].transformation_const().translation());
        m_inertiaInv[iActor] = Util::toSycl(actors[iActor].inertiaInv());
        m_linearVelocity[iActor] = Util::toSycl(actors[iActor].linearVelocity());

        const auto& vertices = actors[iActor].vertexPositionsWorld();
        std::fill(m_actorIndices.begin()+vertexOffset, m_actorIndices.begin()+vertexOffset+vertices[0].size(), iActor);
        std::copy(vertices[0].begin(), vertices[0].end(), m_allVertices[0].begin()+vertexOffset);
        std::copy(vertices[1].begin(), vertices[1].end(), m_allVertices[1].begin()+vertexOffset);
        std::copy(vertices[2].begin(), vertices[2].end(), m_allVertices[2].begin()+vertexOffset);
        vertexOffset += vertices[0].size();
    }
}

// -----------------------------------------------------------------------------
void CollisionSim::State::load(const std::vector<Actor>& actors) {
    size_t vertexOffset{0};
    for (size_t iActor{0}; iActor<m_numActors; ++iActor) {
        m_translation[iActor] = Util::toSycl(actors[iActor].transformation_const().translation());
        m_rotation[iActor] = Util::toSycl(actors[iActor].transformation_const().rotation());
        m_inertiaInv[iActor] = Util::toSycl(actors[iActor].inertiaInv());
        m_linearVelocity[iActor] = Util::toSycl(actors[iActor].linearVelocity());
        m_angularVelocity[iActor] = Util::toSycl(actors[iActor].angularVelocity());
        m_linearMomentum[iActor] = Util::toSycl(actors[iActor].linearMomentum());
        m_angularMomentum[iActor] = Util::toSycl(actors[iActor].angularMomentum());
        m_force[iActor] = Util::toSycl(actors[iActor].force());
        m_torque[iActor] = Util::toSycl(actors[iActor].torque());

        const auto& vertices = actors[iActor].vertexPositionsWorld();
        std::copy(vertices[0].begin(), vertices[0].end(), m_allVertices[0].begin()+vertexOffset);
        std::copy(vertices[1].begin(), vertices[1].end(), m_allVertices[1].begin()+vertexOffset);
        std::copy(vertices[2].begin(), vertices[2].end(), m_allVertices[2].begin()+vertexOffset);
        vertexOffset += vertices[0].size();
    }
}

// -----------------------------------------------------------------------------
void CollisionSim::State::resetBuffers() {
    // NB: Actor indices, mass and body inertia don't change, so we don't reset their buffers
    m_vxBuf = {m_allVertices[0].data(), m_numAllVertices};
    m_vyBuf = {m_allVertices[1].data(), m_numAllVertices};
    m_vzBuf = {m_allVertices[2].data(), m_numAllVertices};
    m_translationBuf = {m_translation.data(), m_numActors};
    m_rotationBuf = {m_rotation.data(), m_numActors};
    m_inertiaInvBuf = {m_inertiaInv.data(), m_numActors};
    m_linearVelocityBuf = {m_linearVelocity.data(), m_numActors};
    m_angularVelocityBuf = {m_linearVelocity.data(), m_numActors};
    m_linearMomentumBuf = {m_linearMomentum.data(), m_numActors};
    m_angularMomentumBuf = {m_linearMomentum.data(), m_numActors};
    m_forceBuf = {m_force.data(), m_numActors};
    m_torqueBuf = {m_torque.data(), m_numActors};
    m_wallCollisionCache.resetBuffers();
}
