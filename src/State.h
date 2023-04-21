/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_STATE
#define COLLISION_SIM_STATE

#include "Actor.h"
#include "Wall.h"
#include <sycl/sycl.hpp>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Range.h>
#include <vector>
#include <array>

namespace CollisionSim {

class WallCollisionCache {
    public:
        WallCollisionCache(const Magnum::Range3D& worldBoundaries, size_t numAllVertices);
        WallCollisionCache(const WallCollisionCache&) = delete;
        WallCollisionCache(WallCollisionCache&&) = delete;
        WallCollisionCache& operator=(const WallCollisionCache&) = delete;
        WallCollisionCache& operator=(WallCollisionCache&&) = delete;
        const std::array<float,6>& boundaries() const {return m_boundaries;}
        const std::vector<Wall>& collisions() const {return m_collisions;}
        const std::vector<sycl::float3>& addLinearVelocity() const {return m_addLinearVelocity;}
        const std::vector<sycl::float3>& addAngularVelocity() const {return m_addAngularVelocity;}
        sycl::buffer<float,1>& boundariesBuf() {return m_boundariesBuf;}
        sycl::buffer<Wall,1>& collisionsBuf() {return m_collisionsBuf;}
        sycl::buffer<sycl::float3,1>& addLinearVelocityBuf() {return m_addLinearVelocityBuf;}
        sycl::buffer<sycl::float3,1>& addAngularVelocityBuf() {return m_addAngularVelocityBuf;}
        void resetBuffers();
    private:
        ///@{ Data
        std::array<float,6> m_boundaries;
        std::vector<Wall> m_collisions;
        std::vector<sycl::float3> m_addLinearVelocity;
        std::vector<sycl::float3> m_addAngularVelocity;
        ///@}

        ///@{ Buffers
        sycl::buffer<float,1> m_boundariesBuf;
        sycl::buffer<Wall,1> m_collisionsBuf;
        sycl::buffer<sycl::float3,1> m_addLinearVelocityBuf;
        sycl::buffer<sycl::float3,1> m_addAngularVelocityBuf;
        ///@}
};

/**
 * Class representing the simulation state, with properties
 * of all actors formatted into contiguous arrays
 */
class State {
    public:
        using float3x3 = std::array<sycl::float3,3>;

        /// Empty constructor
        State() = delete;
        /**
         * Constructor from a vector of actors
         */
        State(const Magnum::Range3D& worldBoundaries, const std::vector<Actor>& actors, size_t numAllVertices);

        /// Copy and assignment explicitly deleted
        ///@{
        State(const State&) = delete;
        State(State&&) = delete;
        State& operator=(const State&) = delete;
        State& operator=(State&&) = delete;
        ///}

        /// Copy data from a vector of actors
        void load(const std::vector<Actor>& actors);

        void resetBuffers();

        /// Const data getters
        ///{@
        size_t numActors() const {return m_numActors;}
        size_t numAllVertices() const {return m_numAllVertices;}
        const std::vector<float>& mass() const {return m_mass;}
        const std::vector<uint16_t>& actorIndices() const {return m_actorIndices;}
        const std::array<std::vector<float>,3>& allVertices() const {return m_allVertices;}
        const std::vector<sycl::float3>& translation() const {return m_translation;}
        const std::vector<float3x3>& inertiaInv() const {return m_inertiaInv;}
        const std::vector<sycl::float3>& linearVelocity() const {return m_linearVelocity;}
        ///@}

        WallCollisionCache& wallCollisionCache() {return m_wallCollisionCache;}

        /// Buffer getters
        ///{@
        sycl::buffer<uint16_t,1>& actorIndicesBuf() {return m_actorIndicesBuf;}
        sycl::buffer<float,1>& massBuf() {return m_massBuf;}
        sycl::buffer<float,1>& vxBuf() {return m_vxBuf;}
        sycl::buffer<float,1>& vyBuf() {return m_vyBuf;}
        sycl::buffer<float,1>& vzBuf() {return m_vzBuf;}
        sycl::buffer<sycl::float3,1>& translationBuf() {return m_translationBuf;}
        sycl::buffer<float3x3,1>& inertiaInvBuf() {return m_inertiaInvBuf;}
        sycl::buffer<sycl::float3,1>& linearVelocityBuf() {return m_linearVelocityBuf;}
        ///@}
    private:
        /// Constants
        ///@{
        size_t m_numActors{0};
        size_t m_numAllVertices{0};
        std::vector<float> m_mass;
        std::vector<uint16_t> m_actorIndices; // Caution: restricting numActors to 65536
        ///@}
        /// Mutable data
        /// @{
        std::array<std::vector<float>,3> m_allVertices;
        std::vector<sycl::float3> m_translation;
        std::vector<float3x3> m_inertiaInv;
        std::vector<sycl::float3> m_linearVelocity;
        WallCollisionCache m_wallCollisionCache;
        ///@}
        /// SYCL buffers
        ///@{
        sycl::buffer<uint16_t,1> m_actorIndicesBuf;
        sycl::buffer<float,1> m_massBuf;
        sycl::buffer<float,1> m_vxBuf;
        sycl::buffer<float,1> m_vyBuf;
        sycl::buffer<float,1> m_vzBuf;
        sycl::buffer<sycl::float3,1> m_translationBuf;
        sycl::buffer<float3x3,1> m_inertiaInvBuf;
        sycl::buffer<sycl::float3,1> m_linearVelocityBuf;
        ///}
};
} // namespace CollisionSim

#endif // COLLISION_SIM_STATE
