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
  m_mass(actors.size(), 0.0f),
  m_actorIndices(numAllVertices, uint16_t{0}),
  m_allVertices{std::vector<float>(numAllVertices, 0.0f), std::vector<float>(numAllVertices, 0.0f), std::vector<float>(numAllVertices, 0.0f)},
  m_translation(actors.size(), sycl::float3{0.0f}),
  m_inertiaInv(actors.size(), {sycl::float3{0.0f},sycl::float3{0.0f},sycl::float3{0.0f}}),
  m_linearVelocity(actors.size(), sycl::float3{0.0f}),
  m_wallCollisionCache{worldBoundaries, numAllVertices},
  m_actorIndicesBuf{m_actorIndices.data(), numAllVertices},
  m_massBuf{m_mass.data(), actors.size()},
  m_vxBuf{m_allVertices[0].data(), numAllVertices},
  m_vyBuf{m_allVertices[1].data(), numAllVertices},
  m_vzBuf{m_allVertices[2].data(), numAllVertices},
  m_translationBuf{m_translation.data(), actors.size()},
  m_inertiaInvBuf{m_inertiaInv.data(), actors.size()},
  m_linearVelocityBuf{m_linearVelocity.data(), actors.size()} {

    size_t vertexOffset{0};
    for (size_t iActor{0}; iActor<m_numActors; ++iActor) {
        m_mass[iActor] = actors[iActor].mass();
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
        m_inertiaInv[iActor] = Util::toSycl(actors[iActor].inertiaInv());
        m_linearVelocity[iActor] = Util::toSycl(actors[iActor].linearVelocity());

        const auto& vertices = actors[iActor].vertexPositionsWorld();
        std::copy(vertices[0].begin(), vertices[0].end(), m_allVertices[0].begin()+vertexOffset);
        std::copy(vertices[1].begin(), vertices[1].end(), m_allVertices[1].begin()+vertexOffset);
        std::copy(vertices[2].begin(), vertices[2].end(), m_allVertices[2].begin()+vertexOffset);
        vertexOffset += vertices[0].size();
    }
}

// -----------------------------------------------------------------------------
void CollisionSim::State::resetBuffers() {
    /* Actor indices and mass don't change, no need to reset
    m_actorIndicesBuf = {m_actorIndices.data(), m_numAllVertices};
    m_massBuf = {m_mass.data(), m_numActors};
    */
    m_vxBuf = {m_allVertices[0].data(), m_numAllVertices};
    m_vyBuf = {m_allVertices[1].data(), m_numAllVertices};
    m_vzBuf = {m_allVertices[2].data(), m_numAllVertices};
    m_translationBuf = {m_translation.data(), m_numActors};
    m_inertiaInvBuf = {m_inertiaInv.data(), m_numActors};
    m_linearVelocityBuf = {m_linearVelocity.data(), m_numActors};
    m_wallCollisionCache.resetBuffers();
}
