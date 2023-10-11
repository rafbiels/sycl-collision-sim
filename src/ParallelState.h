/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_PARALLELSTATE
#define COLLISION_SIM_PARALLELSTATE

#include "Constants.h"
#include "Edge.h"
#include "USMData.h"
#include "Wall.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Range.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <array>
#include <unordered_set>

namespace CollisionSim {

class Actor;

/**
 * Class representing the simulation state for parallel simulation,
 * with properties of all actors formatted into contiguous arrays
 */
class ParallelState {
    public:
        using float3x3 = std::array<sycl::float3,3>;

        /// Empty constructor
        ParallelState() = delete;
        /**
         * Constructor from a vector of actors
         */
        explicit ParallelState(const Magnum::Range3D& worldBounds,
                               const std::vector<Actor>& actors,
                               size_t numAllVertices,
                               size_t numAllTriangles,
                               const sycl::queue& queue);

        /// Copy and assignment explicitly deleted
        ///@{
        ParallelState(const ParallelState&) = delete;
        ParallelState(ParallelState&&) = delete;
        ParallelState& operator=(const ParallelState&) = delete;
        ParallelState& operator=(ParallelState&&) = delete;
        ///}

        /// Enqueue copy of all data to the device and return immediately
        void copyAllToDeviceAsync() const;

        size_t numAllVertices{0};
        size_t maxNumVerticesPerActor{0};

        /// Constants
        ///@{
        USMData<float> worldBoundaries;
        USMData<uint16_t> actorIndices; // Caution: restricting N actors to 65536
        USMData<float,Constants::NumActors> mass;
        USMData<float3x3,Constants::NumActors> bodyInertiaInv;
        USMData<uint16_t,Constants::NumActors> numVertices; // Caution: restricting N vertices per actor to 65536
        USMData<unsigned int,Constants::NumActors> verticesOffset;
        USMData<sycl::float3> bodyVertices;
        USMData<uint16_t,Constants::NumActors> numTriangles; // Caution: restricting N triangles per actor to 65536
        USMData<unsigned int,Constants::NumActors> trianglesOffset;
        USMData<sycl::uint3> triangles;
        ///@}

        /// Motion simulation variables
        ///@{
        USMData<sycl::float3> worldVertices;
        USMData<sycl::float3,Constants::NumActors> translation;
        USMData<float3x3,Constants::NumActors> rotation;
        USMData<float3x3,Constants::NumActors> inertiaInv;
        USMData<sycl::float3,Constants::NumActors> linearVelocity;
        USMData<sycl::float3,Constants::NumActors> angularVelocity;
        USMData<sycl::float3,Constants::NumActors> linearMomentum;
        USMData<sycl::float3,Constants::NumActors> angularMomentum;
        USMData<sycl::float3,Constants::NumActors> force;
        USMData<sycl::float3,Constants::NumActors> torque;
        ///@}

        /// Collision simulation variables
        ///@{
        USMData<Wall> wallCollisions;
        USMData<sycl::float3> addLinearVelocity; /// Per-vertex collision response
        USMData<sycl::float3> addAngularVelocity; /// Per-vertex collision response
        std::array<USMData<sycl::float2, Constants::NumActors>,3> aabb;
        std::array<USMData<Edge, 2*Constants::NumActors>,3> sortedAABBEdges;
        USMData<bool,Constants::NumActorPairs> aabbOverlaps;
        USMData<int,Constants::NumActors> pairedActorIndices;
        USMData<int,Constants::NumActors> actorImpulseApplied; /// Lock to avoid concurrent collisions
        size_t aabbOverlapsLastFrame{0};
        ///@}
};


/**
 * Structure holding the result of triangle-vertex matching
 *
 * {float3 point on triangle, float3 normal, float distance squared}
 */
struct TVMatch {
    std::array<float,3> pointOnTriangle{0.0f};
    std::array<float,3> normal{0.0f};
    float dsq{std::numeric_limits<float>::max()};
};

/**
 * Class representing the narrow phase state with dynamic data size
 * depending on the output of the broad phase collision detection
 */
class NarrowPhaseState {
    public:
        explicit NarrowPhaseState(size_t nTrianglesToCheck,
                                  const std::unordered_set<uint16_t>& overlappingActorSet,
                                  const sycl::queue& queue);
        size_t numTrianglesToCheck{0};
        size_t numActorsToCheck{0};
        USMData<uint16_t> overlappingActors;
        USMData<TVMatch> triangleVertexMatch;
        USMData<TVMatch> triangleBestMatch;
};

} // namespace CollisionSim

#endif // COLLISION_SIM_PARALLELSTATE
