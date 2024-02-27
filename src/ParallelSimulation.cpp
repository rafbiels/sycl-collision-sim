/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "ParallelSimulation.h"
#include "ParallelState.h"
#include "Actor.h"
#include "Constants.h"
#include "Util.h"
#include "Wall.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <sycl/ext/oneapi/experimental/user_defined_reductions.hpp>
#include <sycl/ext/oneapi/experimental/group_helpers_sorters.hpp>
#include <limits>
#include <numeric>
#include <unordered_set>

namespace CollisionSim::Simulation {
using float3x3 = CollisionSim::ParallelState::float3x3;

/// Compute motion step and reset per-actor temporary data
class ActorKernel {
private:
    float m_dtime{0.0f};
    float* m_worldBoundaries{nullptr};
    float* m_mass{nullptr};
    sycl::float3* m_linearVelocity{nullptr};
    float3x3* m_inertiaInv{nullptr};
    sycl::float3* m_translation{nullptr};
    float3x3* m_bodyInertiaInv{nullptr};
    sycl::float3* m_angularVelocity{nullptr};
    float3x3* m_rotation{nullptr};
    sycl::float3* m_force{nullptr};
    sycl::float3* m_torque{nullptr};
    int* m_pairedActorIndices{nullptr};
    int* m_actorImpulseApplied{nullptr};

public:
    constexpr explicit ActorKernel(float dtime, const ParallelState& state) noexcept
    : m_dtime{dtime},
      m_worldBoundaries{state.worldBoundaries.devicePointer},
      m_mass{state.mass.devicePointer},
      m_linearVelocity{state.linearVelocity.devicePointer},
      m_inertiaInv{state.inertiaInv.devicePointer},
      m_translation{state.translation.devicePointer},
      m_bodyInertiaInv{state.bodyInertiaInv.devicePointer},
      m_angularVelocity{state.angularVelocity.devicePointer},
      m_rotation{state.rotation.devicePointer},
      m_force{state.force.devicePointer},
      m_torque{state.torque.devicePointer},
      m_pairedActorIndices{state.pairedActorIndices.devicePointer},
      m_actorImpulseApplied{state.actorImpulseApplied.devicePointer}
    {}

    void operator()(sycl::id<1> id) const {
        // Compute linear and angular momentum
        auto linearMomentum = m_mass[id] * m_linearVelocity[id];
        auto angularMomentum = Util::mvmul(Util::inverse(m_inertiaInv[id]), m_angularVelocity[id]);

        linearMomentum += m_force[id] * m_dtime;
        angularMomentum += m_torque[id] * m_dtime;

        // Compute linear and angular velocity
        m_linearVelocity[id] = linearMomentum / m_mass[id];
        m_inertiaInv[id] = Util::mmul(
            Util::mmul(m_rotation[id], m_bodyInertiaInv[id]),
            Util::transpose(m_rotation[id])); // R * Ib^-1 * R^T
        m_angularVelocity[id] = Util::mvmul(m_inertiaInv[id], angularMomentum);

        // Apply translation
        m_translation[id] += m_linearVelocity[id] * m_dtime;

        // Protect actors from escaping the world
        #pragma unroll
        for (unsigned int axis{0}; axis<3; ++axis) {
            m_translation[id][0] = sycl::max(m_translation[id][0], m_worldBoundaries[0]);
            m_translation[id][0] = sycl::min(m_translation[id][0], m_worldBoundaries[1]);
            m_translation[id][1] = sycl::max(m_translation[id][1], m_worldBoundaries[2]);
            m_translation[id][1] = sycl::min(m_translation[id][1], m_worldBoundaries[3]);
            m_translation[id][2] = sycl::max(m_translation[id][2], m_worldBoundaries[4]);
            m_translation[id][2] = sycl::min(m_translation[id][2], m_worldBoundaries[5]);
        }

        // Apply rotation
        auto star = [](const sycl::float3& v) constexpr {
            return std::array<sycl::float3,3>{
                sycl::float3{ 0.0f,  v[2], -v[1]},
                sycl::float3{-v[2],  0.0f,  v[0]},
                sycl::float3{ v[1], -v[0],  0.0f}
            };
        };
        std::array<sycl::float3,3> drot = Util::msmul(
            Util::mmul(star(m_angularVelocity[id]), m_rotation[id]),
            m_dtime);
        m_rotation[id][0] += drot[0];
        m_rotation[id][1] += drot[1];
        m_rotation[id][2] += drot[2];

        // Reset AABB pairing info and impulse lock
        m_pairedActorIndices[id] = -1;
        m_actorImpulseApplied[id] = 0;
    }
};

/// Update vertex positions and calculate world collisions
class VertexKernel {
private:
    float* m_worldBoundaries{nullptr};
    float* m_mass{nullptr};
    uint16_t* m_actorIndices{nullptr};
    sycl::float3* m_linearVelocity{nullptr};
    float3x3* m_inertiaInv{nullptr};
    sycl::float3* m_translation{nullptr};
    sycl::float3* m_addLinearVelocity{nullptr};
    sycl::float3* m_addAngularVelocity{nullptr};
    Wall* m_wallCollisions{nullptr};
    sycl::float3* m_bodyVertices{nullptr};
    sycl::float3* m_worldVertices{nullptr};
    float3x3* m_rotation{nullptr};

public:
    constexpr explicit VertexKernel(const ParallelState& state) noexcept
    : m_worldBoundaries{state.worldBoundaries.devicePointer},
      m_mass{state.mass.devicePointer},
      m_actorIndices{state.actorIndices.devicePointer},
      m_linearVelocity{state.linearVelocity.devicePointer},
      m_inertiaInv{state.inertiaInv.devicePointer},
      m_translation{state.translation.devicePointer},
      m_addLinearVelocity{state.addLinearVelocity.devicePointer},
      m_addAngularVelocity{state.addAngularVelocity.devicePointer},
      m_wallCollisions{state.wallCollisions.devicePointer},
      m_bodyVertices{state.bodyVertices.devicePointer},
      m_worldVertices{state.worldVertices.devicePointer},
      m_rotation{state.rotation.devicePointer}
    {}

    void operator()(sycl::id<1> id) const {
        uint16_t iActor = m_actorIndices[id];

        m_worldVertices[id] = sycl::float3{
            // x
            m_rotation[iActor][0][0]*m_bodyVertices[id][0] +
            m_rotation[iActor][1][0]*m_bodyVertices[id][1] +
            m_rotation[iActor][2][0]*m_bodyVertices[id][2] +
            m_translation[iActor][0],
            // y
            m_rotation[iActor][0][1]*m_bodyVertices[id][0] +
            m_rotation[iActor][1][1]*m_bodyVertices[id][1] +
            m_rotation[iActor][2][1]*m_bodyVertices[id][2] +
            m_translation[iActor][1],
            // z
            m_rotation[iActor][0][2]*m_bodyVertices[id][0] +
            m_rotation[iActor][1][2]*m_bodyVertices[id][1] +
            m_rotation[iActor][2][2]*m_bodyVertices[id][2] +
            m_translation[iActor][2]
        };

        Wall collision{Wall::None};
        collision |= (static_cast<WallUnderlyingType>(m_worldVertices[id][0] <= m_worldBoundaries[0]) << 0);
        collision |= (static_cast<WallUnderlyingType>(m_worldVertices[id][0] >= m_worldBoundaries[1]) << 1);
        collision |= (static_cast<WallUnderlyingType>(m_worldVertices[id][1] <= m_worldBoundaries[2]) << 2);
        collision |= (static_cast<WallUnderlyingType>(m_worldVertices[id][1] >= m_worldBoundaries[3]) << 3);
        collision |= (static_cast<WallUnderlyingType>(m_worldVertices[id][2] <= m_worldBoundaries[4]) << 4);
        collision |= (static_cast<WallUnderlyingType>(m_worldVertices[id][2] >= m_worldBoundaries[5]) << 5);

        sycl::float3 normal = wallNormal(collision);
        sycl::float3 radius{m_worldVertices[id] - m_translation[iActor]};
        sycl::float3 a{sycl::cross(radius, normal)};
        sycl::float3 b{Util::mvmul(m_inertiaInv[iActor], a)};
        sycl::float3 c{sycl::cross(b, radius)};
        float d{sycl::dot(c, normal)};
        float impulse = (-1.0f - Constants::RestitutionCoefficient) *
                        sycl::dot(m_linearVelocity[iActor], normal) /
                        (1.0f/m_mass[iActor] + d);

        m_addLinearVelocity[id] = (impulse / m_mass[iActor]) * normal;
        m_addAngularVelocity[id] = impulse * b;
        bool ignoreAwayFromWall{sycl::dot(m_linearVelocity[iActor], normal) > 0.0f};
        m_wallCollisions[id] = static_cast<Wall>(
            static_cast<WallUnderlyingType>(collision) *
            static_cast<WallUnderlyingType>(!ignoreAwayFromWall));
    }
};

/// Calculate the axis-align bounding boxes for each actor
///
/// The use of sycl::joint_reduce requires 1D arrays of vx, vy, vz
/// so we copy the AoS global memory vertices into local memory SoA
class AABBKernel {
private:
    uint16_t* m_numVertices{nullptr};
    uint32_t* m_verticesOffset{nullptr};
    sycl::float3* m_worldVertices{nullptr};
    std::array<sycl::float2*,3> m_aabb{nullptr};
    std::array<sycl::local_accessor<float,1>,3> m_localVertices{};

public:
    constexpr static size_t workGroupSize{32};

    explicit AABBKernel(sycl::handler& cgh, const ParallelState& state) noexcept
    : m_numVertices{state.numVertices.devicePointer},
      m_verticesOffset{state.verticesOffset.devicePointer},
      m_worldVertices{state.worldVertices.devicePointer},
      m_aabb{state.aabb[0].devicePointer, state.aabb[1].devicePointer, state.aabb[2].devicePointer},
      m_localVertices{sycl::local_accessor<float,1>{sycl::range<1>{state.maxNumVerticesPerActor},cgh},
                      sycl::local_accessor<float,1>{sycl::range<1>{state.maxNumVerticesPerActor},cgh},
                      sycl::local_accessor<float,1>{sycl::range<1>{state.maxNumVerticesPerActor},cgh}}
    {}

    void operator()(sycl::nd_item<1> item) const {
        size_t iActor = item.get_group_linear_id();

        sycl::float3* actorVertices = m_worldVertices + m_verticesOffset[iActor];
        size_t numVerticesPerThread{1 + m_numVertices[iActor]/workGroupSize};
        for (size_t i{0}; i<numVerticesPerThread; ++i) {
            size_t iVertex{item.get_local_linear_id() * numVerticesPerThread + i};
            if (iVertex<m_numVertices[iActor]) {
                m_localVertices[0][iVertex] = actorVertices[iVertex][0];
                m_localVertices[1][iVertex] = actorVertices[iVertex][1];
                m_localVertices[2][iVertex] = actorVertices[iVertex][2];
            }
        }

        sycl::group_barrier(item.get_group());

        #pragma unroll
        for (unsigned int axis{0}; axis<3; ++axis) {
            m_aabb[axis][iActor] = sycl::float2{
                sycl::joint_reduce(
                    item.get_group(),
                    m_localVertices[axis].begin(),
                    m_localVertices[axis].begin()+m_numVertices[iActor],
                    sycl::minimum{}),
                sycl::joint_reduce(
                    item.get_group(),
                    m_localVertices[axis].begin(),
                    m_localVertices[axis].begin()+m_numVertices[iActor],
                    sycl::maximum{})
            };
        }
    }
};

/// Sort the AABB edges using odd-even merge-sort
///
/// Given the small size of the problem we can avoid submitting N(=2*NumActors) kernels.
/// Instead, we can submit one kernel with a single work-group per axis and exploit a work-group barrier.
/// This requires that N is smaller than the maximum work-group size of the GPU we're using.
class AABBSortKernel {
private:
    std::array<sycl::float2*,3> m_aabb{nullptr};
    std::array<Edge*,3> m_sortedAABBEdges{nullptr};

public:
    constexpr explicit AABBSortKernel(const ParallelState& state) noexcept
    : m_aabb{state.aabb[0].devicePointer,
             state.aabb[1].devicePointer,
             state.aabb[2].devicePointer},
      m_sortedAABBEdges{state.sortedAABBEdges[0].devicePointer,
                        state.sortedAABBEdges[1].devicePointer,
                        state.sortedAABBEdges[2].devicePointer}
    {}

    void operator()(sycl::nd_item<1> item) const {
        size_t axis = item.get_group_linear_id();
        size_t id = item.get_local_linear_id();
        auto edgeValue = [&axis, this](Edge e) constexpr -> float {
            return e.isEnd ? m_aabb[axis][e.actorIndex][1] : m_aabb[axis][e.actorIndex][0];
        };
        auto edgeGreater = [&edgeValue](Edge edgeA, Edge edgeB) constexpr -> bool {
            return edgeValue(edgeA) > edgeValue(edgeB);
        };
        auto compareExchange = [&edgeGreater](Edge& edgeA, Edge& edgeB) constexpr -> void {
            if (edgeGreater(edgeA,edgeB)) {std::swap(edgeA, edgeB);}
        };
        for (size_t step{0}; step < 2*Constants::NumActors; ++step) {
            size_t i = id * 2 + step%2;
            if ((i+1) < (2*Constants::NumActors)) {
                compareExchange(m_sortedAABBEdges[axis][i], m_sortedAABBEdges[axis][i+1]);
            }
            sycl::group_barrier(item.get_group());
        }
    }
};

/// Find overlapping AABB pairs
class AABBOverlapKernel {
private:
    std::array<Edge*,3> m_sortedAABBEdges{nullptr};
    bool* m_aabbOverlaps{nullptr};
    int* m_pairedActorIndices{nullptr};

public:
    constexpr explicit AABBOverlapKernel(const ParallelState& state) noexcept
    : m_sortedAABBEdges{state.sortedAABBEdges[0].devicePointer,
                        state.sortedAABBEdges[1].devicePointer,
                        state.sortedAABBEdges[2].devicePointer},
      m_aabbOverlaps{state.aabbOverlaps.devicePointer},
      m_pairedActorIndices{state.pairedActorIndices.devicePointer}
    {}

    void operator()(sycl::id<1> id) const {
        size_t iActorA{Constants::ActorPairs[id].first};
        size_t iActorB{Constants::ActorPairs[id].second};
        sycl::int3 posStartA{-1};
        sycl::int3 posStartB{-1};
        sycl::int3 posEndA{-1};
        sycl::int3 posEndB{-1};
        for (int iEdge{0}; iEdge<static_cast<int>(2*Constants::NumActors); ++iEdge) {
            #pragma unroll
            for (int axis{0}; axis<3; ++axis) {
                const Edge& edge{m_sortedAABBEdges[axis][iEdge]};
                if (edge.actorIndex==iActorA) {
                    if (edge.isEnd) {posEndA[axis] = iEdge;}
                    else {posStartA[axis] = iEdge;}
                } else if (edge.actorIndex==iActorB) {
                    if (edge.isEnd) {posEndB[axis] = iEdge;}
                    else {posStartB[axis] = iEdge;}
                }
            }
        }
        auto overlap = [](int a1, int a2, int b1, int b2) constexpr -> bool {
            return (a1 < b1 && b1 < a2) || (b1 < a1 && a1 < b2);
        };
        m_aabbOverlaps[id] = (
            overlap(posStartA[0], posEndA[0], posStartB[0], posEndB[0]) &&
            overlap(posStartA[1], posEndA[1], posStartB[1], posEndB[1]) &&
            overlap(posStartA[2], posEndA[2], posStartB[2], posEndB[2])
        );
        if (m_aabbOverlaps[id]) {
            m_pairedActorIndices[iActorA] = static_cast<int>(iActorB);
            m_pairedActorIndices[iActorB] = static_cast<int>(iActorA);
        }
    }
};

/// For each triangle, find the closest vertex from another actor
class NarrowPhaseKernel {
private:
    size_t m_numTrianglesToCheck{0};
    size_t m_numActorsToCheck{0};
    uint16_t* m_overlappingActors{nullptr};
    TVMatch* m_triangleVertexMatch{nullptr};
    sycl::float3* m_translation{nullptr};
    uint16_t* m_numVertices{nullptr};
    uint32_t* m_verticesOffset{nullptr};
    sycl::float3* m_worldVertices{nullptr};
    uint16_t* m_numTriangles{nullptr};
    uint32_t* m_trianglesOffset{nullptr};
    sycl::uint3* m_triangles{nullptr};
    int* m_pairedActorIndices{nullptr};

public:
    constexpr explicit NarrowPhaseKernel(const NarrowPhaseState& npState, const ParallelState& state) noexcept
    : m_numTrianglesToCheck{npState.numTrianglesToCheck},
      m_numActorsToCheck{npState.numActorsToCheck},
      m_overlappingActors{npState.overlappingActors.devicePointer},
      m_triangleVertexMatch{npState.triangleVertexMatch.devicePointer},
      m_translation{state.translation.devicePointer},
      m_numVertices{state.numVertices.devicePointer},
      m_verticesOffset{state.verticesOffset.devicePointer},
      m_worldVertices{state.worldVertices.devicePointer},
      m_numTriangles{state.numTriangles.devicePointer},
      m_trianglesOffset{state.trianglesOffset.devicePointer},
      m_triangles{state.triangles.devicePointer},
      m_pairedActorIndices{state.pairedActorIndices.devicePointer}
    {}

    void operator()(sycl::nd_item<1> item) const {
        const unsigned int id{static_cast<unsigned int>(item.get_global_linear_id())};
        if (id < m_numTrianglesToCheck) {
            unsigned int iTriangle{0};
            unsigned int iActor{0};
            unsigned int result_index{0};
            unsigned int offset{0};
            for (unsigned int iThisOverlap{0}; iThisOverlap<m_numActorsToCheck; ++iThisOverlap) {
                const unsigned int iThisActor = m_overlappingActors[iThisOverlap];
                const uint16_t numTrianglesThisActor = m_numTriangles[iThisActor];
                const unsigned int iTriangleThisActor{id-offset};
                if (iTriangleThisActor < numTrianglesThisActor) {
                    iTriangle = m_trianglesOffset[iThisActor] + iTriangleThisActor;
                    iActor = iThisActor;
                    result_index = iThisOverlap*Constants::MaxNumTriangles + iTriangleThisActor;
                }
                offset += numTrianglesThisActor;
            }
            const unsigned int vertexOffset = m_verticesOffset[iActor];
            const sycl::uint3& triangleIndices{m_triangles[iTriangle]};
            std::array<sycl::float3,3> triangle{
                m_worldVertices[vertexOffset+triangleIndices[0]],
                m_worldVertices[vertexOffset+triangleIndices[1]],
                m_worldVertices[vertexOffset+triangleIndices[2]]
            };
            const auto transformed = Util::triangleTransform(triangle);
            const auto& rot = transformed[1];
            const auto& negRot = transformed[2];

            const unsigned int iOtherActor{static_cast<unsigned int>(m_pairedActorIndices[iActor])};
            const unsigned int vertexOffsetOtherActor = m_verticesOffset[iOtherActor];
            const unsigned int numVerticesOtherActor = m_numVertices[iOtherActor];

            sycl::float3 bestPointOnTriangle{0.0f};
            float bestDsq{std::numeric_limits<float>::max()};

            for (unsigned int iVertex{0}; iVertex<numVerticesOtherActor; ++iVertex) {
                sycl::float3 P{m_worldVertices[vertexOffsetOtherActor+iVertex]};
                P -= triangle[0];
                P = Util::mvmul(rot,P);

                const auto [closestPoint, distanceSquared] = Util::closestPointOnTriangle(transformed[0], P);

                if (distanceSquared < bestDsq) {
                    bestDsq = distanceSquared;
                    bestPointOnTriangle = closestPoint;
                }
            }
            bestPointOnTriangle = Util::mvmul(negRot, bestPointOnTriangle) + triangle[0];
            sycl::float3 normal{sycl::cross(triangle[1]-triangle[0], triangle[2]-triangle[0])};
            float normalisation{sycl::length(normal)};
            if (normalisation==0) {
                // Degenerate triangle - make sure it doesn't make it down the computation
                // It is not skipped earlier to avoid thread divergence
                bestDsq = std::numeric_limits<float>::max();
            } else {
                normal /= normalisation;
            }
            sycl::float3 radius{bestPointOnTriangle - m_translation[iActor]};
            float direction = (sycl::dot(normal, radius) < 0) ? -1.0f : 1.0f;
            normal *= direction;

            m_triangleVertexMatch[result_index] = {
                Util::toArray(bestPointOnTriangle),
                Util::toArray(normal),
                bestDsq
            };
        }
    }
};

/// For each colliding actor, reduce all triangle-vertex pairs to the one with the smallest distance
class TVReduceKernel {
private:
    struct TVMatchReduce {
        TVMatch operator()(const TVMatch& a, const TVMatch& b) const {
            return a.dsq<b.dsq ? a : b;
        }
    };

    TVMatch* m_triangleVertexMatch{nullptr};
    TVMatch* m_triangleBestMatch{nullptr};
    sycl::local_accessor<std::byte, 1> m_scratch{};

public:
    constexpr static size_t TempMemorySize{Constants::MaxNumTriangles*sizeof(TVMatch)};

    explicit TVReduceKernel(const NarrowPhaseState& npState, sycl::handler& cgh) noexcept
    : m_triangleVertexMatch{npState.triangleVertexMatch.devicePointer},
      m_triangleBestMatch{npState.triangleBestMatch.devicePointer},
      m_scratch{TempMemorySize, cgh}
    {}

    void operator()(sycl::nd_item<1> item) const {
        const auto groupId{item.get_group_linear_id()};
        TVMatch* start = m_triangleVertexMatch + groupId*Constants::MaxNumTriangles;
        TVMatch* end = start + Constants::MaxNumTriangles;
        sycl::ext::oneapi::experimental::group_with_scratchpad handle{
            item.get_group(), sycl::span{&m_scratch[0], TempMemorySize}};
        m_triangleBestMatch[groupId] =
            sycl::ext::oneapi::experimental::joint_reduce(handle, start, end, TVMatchReduce{});
    }
};

/// Apply the impulse force to colliding actors
class ImpulseCollisionKernel {
private:
    size_t m_numActorsToCheck{0};
    uint16_t* m_overlappingActors{nullptr};
    TVMatch* m_triangleBestMatch{nullptr};
    float* m_mass{nullptr};
    sycl::float3* m_linearVelocity{nullptr};
    float3x3* m_inertiaInv{nullptr};
    sycl::float3* m_translation{nullptr};
    sycl::float3* m_angularVelocity{nullptr};
    int* m_pairedActorIndices{nullptr};
    int* m_actorImpulseApplied{nullptr};

public:
    constexpr explicit ImpulseCollisionKernel(const NarrowPhaseState& npState, const ParallelState& state) noexcept
    : m_numActorsToCheck{npState.numActorsToCheck},
      m_overlappingActors{npState.overlappingActors.devicePointer},
      m_triangleBestMatch{npState.triangleBestMatch.devicePointer},
      m_mass{state.mass.devicePointer},
      m_linearVelocity{state.linearVelocity.devicePointer},
      m_inertiaInv{state.inertiaInv.devicePointer},
      m_translation{state.translation.devicePointer},
      m_angularVelocity{state.angularVelocity.devicePointer},
      m_pairedActorIndices{state.pairedActorIndices.devicePointer},
      m_actorImpulseApplied{state.actorImpulseApplied.devicePointer}
    {}

    void operator()(sycl::id<1> id) const {
        uint16_t iActorA = m_overlappingActors[id];
        uint16_t iActorB = m_pairedActorIndices[iActorA];
        unsigned int idB{std::numeric_limits<unsigned int>::max()};
        for (unsigned int i{0}; i<static_cast<unsigned int>(m_numActorsToCheck); ++i) {
            if (m_overlappingActors[i]==iActorB) {idB = i;}
        }
        const TVMatch& tvA = m_triangleBestMatch[id];
        const TVMatch& tvB = m_triangleBestMatch[idB];
        if (tvA.dsq < Constants::NarrowPhaseCollisionThreshold && tvA.dsq < tvB.dsq) {
            const sycl::float3& point{Util::toSycl(tvA.pointOnTriangle)};
            const sycl::float3& normal{Util::toSycl(tvA.normal)};
            sycl::float3 ra = point - m_translation[iActorA];
            sycl::float3 rb = point - m_translation[iActorB];
            sycl::float3 vpa = m_linearVelocity[iActorA] + sycl::cross(m_angularVelocity[iActorA], ra);
            sycl::float3 vpb = m_linearVelocity[iActorB] + sycl::cross(m_angularVelocity[iActorB], rb);
            sycl::float3 vr = vpb - vpa;
            sycl::float3 ta = Util::mvmul(m_inertiaInv[iActorA], sycl::cross(ra,normal));
            sycl::float3 tb = Util::mvmul(m_inertiaInv[iActorB], sycl::cross(rb,normal));
            float impulse =
                (-1.0f - Constants::RestitutionCoefficient) *
                sycl::dot(vr,normal) / (
                    1.0f/m_mass[iActorA] +
                    1.0f/m_mass[iActorB] +
                    sycl::dot(
                        sycl::cross(ta, ra) +
                        sycl::cross(tb, rb)
                        , normal
                    )
                );
            sycl::float3 addLinVA = -1.0f * normal * impulse / m_mass[iActorA];
            sycl::float3 addLinVB = normal * impulse / m_mass[iActorB];
            sycl::float3 addAngVA = -1.0f * impulse * ta;
            sycl::float3 addAngVB = impulse * tb;

            auto canApplyImpulse = [this](uint16_t iActor){
                sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::global_space>
                    lock{m_actorImpulseApplied[iActor]};
                int alreadyApplied{0};
                return lock.compare_exchange_strong(alreadyApplied, 1, sycl::memory_order::acq_rel);
            };
            if (canApplyImpulse(iActorA)) {
                m_linearVelocity[iActorA] += addLinVA;
                m_angularVelocity[iActorA] += addAngVA;
            }
            if (canApplyImpulse(iActorB)) {
                m_linearVelocity[iActorB] += addLinVB;
                m_angularVelocity[iActorB] += addAngVB;
            }
        }
    }
};

// -----------------------------------------------------------------------------
void simulateParallel(float dtime, std::vector<Actor>& actors, ParallelState& state, sycl::queue& queue) {

    // Reset collision counter
    state.aabbOverlapsLastFrame = 0;

    // Copy inputs from Actor objects to serial state data
    for (size_t iActor{0}; iActor<Constants::NumActors; ++iActor) {
        state.linearVelocity.hostContainer[iActor] = Util::toSycl(actors[iActor].linearVelocity());
        state.angularVelocity.hostContainer[iActor] = Util::toSycl(actors[iActor].angularVelocity());
        state.force.hostContainer[iActor] = Util::toSycl(actors[iActor].force());
        state.torque.hostContainer[iActor] = Util::toSycl(actors[iActor].torque());
    }
    std::vector<sycl::event> d2hCopyTransformAndVelocity;
    std::vector<sycl::event> d2hCopyWallCollisionInfo;
    try {
        // Copy data modified on host to the device
        std::vector<sycl::event> h2dCopyEvents{
            state.linearVelocity.copyToDevice(),
            state.angularVelocity.copyToDevice(),
            state.force.copyToDevice(),
            state.torque.copyToDevice()
        };

        // Launch 4 kernels to execute motion of actors and their axis-aligned bounding boxes
        sycl::event actorKernelEvent = queue.submit([&h2dCopyEvents, &dtime, &state](sycl::handler& cgh){
            cgh.depends_on(h2dCopyEvents);
            cgh.parallel_for(Constants::NumActors, ActorKernel{dtime,state});
        });

        sycl::event vertexKernelEvent = queue.submit([&actorKernelEvent, &state](sycl::handler& cgh){
            cgh.depends_on(actorKernelEvent);
            cgh.parallel_for(state.numAllVertices, VertexKernel{state});
        });

        sycl::event aabbKernelEvent = queue.submit([&vertexKernelEvent, &state](sycl::handler& cgh){
            cgh.depends_on(vertexKernelEvent);
            const sycl::nd_range<1> aabbRange{Constants::NumActors*AABBKernel::workGroupSize,AABBKernel::workGroupSize};
            cgh.parallel_for(aabbRange, AABBKernel{cgh,state});
        });

        sycl::event aabbSortKernelEvent = queue.submit([&aabbKernelEvent, &state](sycl::handler& cgh){
            cgh.depends_on(aabbKernelEvent);
            const sycl::nd_range<1> aabbSortRange{3*Constants::NumActors, Constants::NumActors};
            cgh.parallel_for(aabbSortRange, AABBSortKernel{state});
        });

        // Start copying wall collision info to the host now, such that the memcpy
        // API calls happen in the shadow of AABBSortKernel still running, and the
        // async copy may continue while AABBOverlapKernel runs
        d2hCopyWallCollisionInfo.insert(d2hCopyWallCollisionInfo.end(),{
            state.wallCollisions.copyToHost(vertexKernelEvent),
            state.addLinearVelocity.copyToHost(vertexKernelEvent),
            state.addAngularVelocity.copyToHost(vertexKernelEvent),
        });

        // Find overlapping AABB pairs
        sycl::event aabbOverlapKernelEvent = queue.submit([&aabbSortKernelEvent, &state](sycl::handler& cgh){
            cgh.depends_on(aabbSortKernelEvent);
            cgh.parallel_for(Constants::NumActorPairs, AABBOverlapKernel{state});
        });

        // Start copying transform info to the host now, such that the memcpy
        // API calls and the copy itself happen in the shadow of the
        // AABBOverlapKernel still running
        d2hCopyTransformAndVelocity.insert(d2hCopyTransformAndVelocity.end(), {
            state.translation.copyToHost(actorKernelEvent),
            state.rotation.copyToHost(actorKernelEvent),
        });

        // Copy back overlap check ouput to host in order to find out how many narrow-phase kernels to submit
        state.pairedActorIndices.copyToHost(aabbOverlapKernelEvent).wait_and_throw();
        std::unordered_set<uint16_t> overlappingActors;
        size_t numTrianglesToCheck{0};
        for (size_t iPair{0}; iPair<Constants::NumActorPairs; ++iPair) {
            const std::pair<int,int>& pair{Constants::ActorPairs[iPair]};
            if (state.pairedActorIndices.hostContainer[pair.first]==pair.second ||
                state.pairedActorIndices.hostContainer[pair.second]==pair.first) {
                ++state.aabbOverlapsLastFrame;
                for (int iActor : {pair.first, pair.second}) {
                    if (overlappingActors.insert(iActor).second) {
                        numTrianglesToCheck += state.numTriangles.hostContainer[iActor];
                    };
                }
            }
        }
        std::optional<sycl::event> impulseCollisionKernelEvent;
        if (!overlappingActors.empty()) {
            // Allocate the narrow-phase state data and copy the overlapping actors information to device
            NarrowPhaseState npState{numTrianglesToCheck, overlappingActors, queue};
            sycl::event copyOverlappingActorsEvent = npState.overlappingActors.copyToDevice();

            // Reset the triangle-vertex match data on the device
            sycl::event resetTriangleVertexMatchEvent = queue.fill(npState.triangleVertexMatch.devicePointer, TVMatch{}, npState.triangleVertexMatch.size());

            // Submit the triangle-vertex matching kernel
            const sycl::nd_range<1> narrowPhaseRange{Util::ndRangeAllCU(npState.numTrianglesToCheck,queue)};
            sycl::event narrowPhaseKernelEvent = queue.submit([&copyOverlappingActorsEvent, &resetTriangleVertexMatchEvent, &narrowPhaseRange, &npState, &state](sycl::handler& cgh){
                cgh.depends_on({copyOverlappingActorsEvent,resetTriangleVertexMatchEvent});
                cgh.parallel_for(narrowPhaseRange, NarrowPhaseKernel{npState, state});
            });

            // For each actor out of numActorsToCheck, reduce all triangle-vertex pairs to the one with the smallest distance
            sycl::event triangleVertexReduceKernelEvent = queue.submit([&narrowPhaseKernelEvent, &npState](sycl::handler& cgh){
                cgh.depends_on(narrowPhaseKernelEvent);
                const sycl::nd_range<1> reduceRange{npState.numActorsToCheck*Constants::MaxNumTriangles, Constants::MaxNumTriangles};
                cgh.parallel_for(reduceRange, TVReduceKernel{npState, cgh});
            });

            // Submit the impulse collision kernel
            impulseCollisionKernelEvent = queue.submit([&triangleVertexReduceKernelEvent, &npState, &state](sycl::handler& cgh){
                cgh.depends_on(triangleVertexReduceKernelEvent);
                cgh.parallel_for(npState.numActorsToCheck, ImpulseCollisionKernel{npState, state});
            });
        }

        // The velocity info copy can start only now, after it is potentially updated
        // in the impulse_collision_kernel
        d2hCopyTransformAndVelocity.insert(d2hCopyTransformAndVelocity.end(), {
            state.linearVelocity.copyToHost(impulseCollisionKernelEvent.value_or(actorKernelEvent)),
            state.angularVelocity.copyToHost(impulseCollisionKernelEvent.value_or(actorKernelEvent)),
        });
    } catch (const std::exception& ex) {
        Corrade::Utility::Error{} << "Exception caught: " << ex.what();
    }

    // Reset force and torque, and transfer serial state data to Actor objects
    try {
        sycl::event::wait_and_throw(d2hCopyTransformAndVelocity);
    } catch (const std::exception& ex) {
        Corrade::Utility::Error{} << "Exception caught: " << ex.what();
    }
    for (size_t iActor{0}; iActor<Constants::NumActors; ++iActor) {
        actors[iActor].force({0, 0, 0});
        actors[iActor].torque({0, 0, 0});
        actors[iActor].transformation(Util::transformationMatrix(state.translation.hostContainer[iActor], state.rotation.hostContainer[iActor]));
        actors[iActor].linearVelocity(Util::toMagnum(state.linearVelocity.hostContainer[iActor]));
        actors[iActor].angularVelocity(Util::toMagnum(state.angularVelocity.hostContainer[iActor]));
        // Fix floating point loss of orthogonality in the rotation matrix
        Util::orthonormaliseRotation(actors[iActor].transformation());
    }

    // Reduce per-vertex world collision outputs to per-actor values and apply actor velocity change
    struct CollisionData {
        Wall type{Wall::None};
        std::vector<sycl::float3> addLinearV;
        std::vector<sycl::float3> addAngularV;
    };
    std::unordered_map<size_t, CollisionData> actorCollisions; // {actor index, collision data}
    try {
        sycl::event::wait_and_throw(d2hCopyWallCollisionInfo);
    } catch (const std::exception& ex) {
        Corrade::Utility::Error{} << "Exception caught: " << ex.what();
    }
    for (size_t iVertex{0}; iVertex<state.numAllVertices; ++iVertex) {
        Wall collision = state.wallCollisions.hostContainer[iVertex];
        if (collision==Wall::None) {continue;}
        CollisionData& data = actorCollisions[state.actorIndices.hostContainer[iVertex]];
        data.type |= collision;
        data.addLinearV.push_back(state.addLinearVelocity.hostContainer[iVertex]);
        data.addAngularV.push_back(state.addAngularVelocity.hostContainer[iVertex]);
    }
    for (const auto& [iActor, data] : actorCollisions) {
        size_t num{data.addLinearV.size()};
        auto accumulateMean = [&num](const sycl::float3& a, const sycl::float3& b){
            return a + b/static_cast<float>(num);
        };
        sycl::float3 meanAddLinV = std::accumulate(data.addLinearV.begin(),data.addLinearV.end(),sycl::float3{0.0f},accumulateMean);
        sycl::float3 meanAddAngV = std::accumulate(data.addAngularV.begin(),data.addAngularV.end(),sycl::float3{0.0f},accumulateMean);

        Magnum::Vector3 addLinV = Util::toMagnum(meanAddLinV);
        // FIXME: why does this happen? fix the logic to avoid this situation
        if (Magnum::Math::dot(actors[iActor].linearVelocity(),addLinV) > 0.0f) {
            continue;
        }
        actors[iActor].addVelocity(addLinV, Util::toMagnum(meanAddAngV));

        const float vy{actors[iActor].linearVelocity().y()};
        // TODO: implement better resting condition
        if ((data.type & Wall::Ymin) > 0 && vy > 0 && vy < 0.01) {
            actors[iActor].addVelocity({0.0f, 0.0001f-1.0f*vy, 0.0f}, {0.0f, 0.0f, 0.0f});
        }
    }
}

} // namespace CollisionSim::Simulation
