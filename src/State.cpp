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
  m_verticesBody{std::vector<float>(numAllVertices, 0.0f), std::vector<float>(numAllVertices, 0.0f), std::vector<float>(numAllVertices, 0.0f)},
  m_verticesWorld{std::vector<float>(numAllVertices, 0.0f), std::vector<float>(numAllVertices, 0.0f), std::vector<float>(numAllVertices, 0.0f)},
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
  m_vxBodyBuf{m_verticesBody[0].data(), numAllVertices},
  m_vyBodyBuf{m_verticesBody[1].data(), numAllVertices},
  m_vzBodyBuf{m_verticesBody[2].data(), numAllVertices},
  m_vxBuf{m_verticesWorld[0].data(), numAllVertices, sycl::property::buffer::use_host_ptr()},
  m_vyBuf{m_verticesWorld[1].data(), numAllVertices, sycl::property::buffer::use_host_ptr()},
  m_vzBuf{m_verticesWorld[2].data(), numAllVertices, sycl::property::buffer::use_host_ptr()},
  m_translationBuf{m_translation.data(), actors.size(), sycl::property::buffer::use_host_ptr()},
  m_rotationBuf{m_rotation.data(), actors.size(), sycl::property::buffer::use_host_ptr()},
  m_inertiaInvBuf{m_inertiaInv.data(), actors.size(), sycl::property::buffer::use_host_ptr()},
  m_linearVelocityBuf{m_linearVelocity.data(), actors.size(), sycl::property::buffer::use_host_ptr()},
  m_angularVelocityBuf{m_angularVelocity.data(), actors.size(), sycl::property::buffer::use_host_ptr()},
  m_linearMomentumBuf{m_linearMomentum.data(), actors.size(), sycl::property::buffer::use_host_ptr()},
  m_angularMomentumBuf{m_angularMomentum.data(), actors.size(), sycl::property::buffer::use_host_ptr()},
  m_forceBuf{m_force.data(), actors.size(), sycl::property::buffer::use_host_ptr()},
  m_torqueBuf{m_torque.data(), actors.size(), sycl::property::buffer::use_host_ptr()} {

    size_t vertexOffset{0};
    for (size_t iActor{0}; iActor<m_numActors; ++iActor) {
        m_mass[iActor] = actors[iActor].mass();
        m_bodyInertiaInv[iActor] = Util::toSycl(actors[iActor].bodyInertiaInv());
        m_translation[iActor] = Util::toSycl(actors[iActor].transformation_const().translation());
        m_inertiaInv[iActor] = Util::toSycl(actors[iActor].inertiaInv());
        m_linearVelocity[iActor] = Util::toSycl(actors[iActor].linearVelocity());

        const auto& verticesBody = actors[iActor].vertexPositions();
        const auto& verticesWorld = actors[iActor].vertexPositionsWorld();
        const size_t numVerticesThisActor{verticesBody[0].size()};
        std::fill(m_actorIndices.begin()+vertexOffset, m_actorIndices.begin()+vertexOffset+numVerticesThisActor, iActor);
        std::copy(verticesBody[0].begin(), verticesBody[0].end(), m_verticesBody[0].begin()+vertexOffset);
        std::copy(verticesBody[1].begin(), verticesBody[1].end(), m_verticesBody[1].begin()+vertexOffset);
        std::copy(verticesBody[2].begin(), verticesBody[2].end(), m_verticesBody[2].begin()+vertexOffset);
        std::copy(verticesWorld[0].begin(), verticesWorld[0].end(), m_verticesWorld[0].begin()+vertexOffset);
        std::copy(verticesWorld[1].begin(), verticesWorld[1].end(), m_verticesWorld[1].begin()+vertexOffset);
        std::copy(verticesWorld[2].begin(), verticesWorld[2].end(), m_verticesWorld[2].begin()+vertexOffset);
        vertexOffset += numVerticesThisActor;
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
        const size_t numVerticesThisActor{vertices[0].size()};
        std::copy(vertices[0].begin(), vertices[0].end(), m_verticesWorld[0].begin()+vertexOffset);
        std::copy(vertices[1].begin(), vertices[1].end(), m_verticesWorld[1].begin()+vertexOffset);
        std::copy(vertices[2].begin(), vertices[2].end(), m_verticesWorld[2].begin()+vertexOffset);
        vertexOffset += numVerticesThisActor;
    }
}

// -----------------------------------------------------------------------------
void CollisionSim::State::store(std::vector<Actor>& actors) const {
    size_t vertexOffset{0};
    for (size_t iActor{0}; iActor<m_numActors; ++iActor) {
        actors[iActor].transformation(Util::transformationMatrix(m_translation[iActor], m_rotation[iActor]));
        actors[iActor].inertiaInv(Util::toMagnum(m_inertiaInv[iActor]));
        actors[iActor].linearVelocity(Util::toMagnum(m_linearVelocity[iActor]));
        actors[iActor].angularVelocity(Util::toMagnum(m_angularVelocity[iActor]));
        actors[iActor].linearMomentum(Util::toMagnum(m_linearMomentum[iActor]));
        actors[iActor].angularMomentum(Util::toMagnum(m_angularMomentum[iActor]));

        const size_t numVerticesThisActor{actors[iActor].vertexPositionsWorld()[0].size()};
        std::copy(m_verticesWorld[0].begin()+vertexOffset,
                  m_verticesWorld[0].begin()+vertexOffset+numVerticesThisActor,
                  actors[iActor].vertexPositionsWorld_nonconst()[0].begin());
        std::copy(m_verticesWorld[1].begin()+vertexOffset,
                  m_verticesWorld[1].begin()+vertexOffset+numVerticesThisActor,
                  actors[iActor].vertexPositionsWorld_nonconst()[1].begin());
        std::copy(m_verticesWorld[2].begin()+vertexOffset,
                  m_verticesWorld[2].begin()+vertexOffset+numVerticesThisActor,
                  actors[iActor].vertexPositionsWorld_nonconst()[2].begin());
        vertexOffset += numVerticesThisActor;
    }
}

// -----------------------------------------------------------------------------
void CollisionSim::State::resetBuffers() {
    // NB: Actor indices, mass, body inertia and internal vertex positions
    // don't change, so we don't reset their buffers
    m_vxBuf = {m_verticesWorld[0].data(), m_numAllVertices};
    m_vyBuf = {m_verticesWorld[1].data(), m_numAllVertices};
    m_vzBuf = {m_verticesWorld[2].data(), m_numAllVertices};
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
