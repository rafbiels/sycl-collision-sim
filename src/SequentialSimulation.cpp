/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "SequentialSimulation.h"
#include "Actor.h"
#include "Constants.h"
#include "Util.h"
#include "Wall.h"
#include <limits>
#include <unordered_set>
#include <utility>

namespace CollisionSim::Simulation {
namespace detail {
// -----------------------------------------------------------------------------
void simulateMotionSequential(float dtime, std::vector<Actor>& actors, const Magnum::Range3D& worldBoundaries) {
    for (auto& actor : actors) {
        // ===========================================
        // Rigid body physics simulation based on D. Baraff 2001
        // https://graphics.pixar.com/pbm2001/pdf/notesg.pdf
        // ===========================================
        // Compute linear and angular momentum
        actor.linearMomentum(actor.linearMomentum() + actor.force() * dtime);
        actor.angularMomentum(actor.angularMomentum() + actor.torque() * dtime);

        // Compute linear and angular velocity
        Magnum::Matrix3 rotation{actor.transformation().rotation()};
        actor.linearVelocity(actor.linearMomentum() / actor.mass());
        actor.inertiaInv(rotation * actor.bodyInertiaInv() * rotation.transposed());
        actor.angularVelocity( actor.inertiaInv() * actor.angularMomentum());

        // Apply translation and rotation
        auto star = [](const Magnum::Vector3& v) {
            return Magnum::Matrix3{
                { 0.0f,  v[2], -v[1]},
                {-v[2],  0.0f,  v[0]},
                { v[1], -v[0],  0.0f}
            };
        };
        Magnum::Matrix3 drot = star(actor.angularVelocity()) * rotation * dtime;
        Magnum::Vector3 dx = actor.linearVelocity() * dtime;


        Magnum::Matrix4 trf{
            {drot[0][0], drot[0][1], drot[0][2], 0.0f},
            {drot[1][0], drot[1][1], drot[1][2], 0.0f},
            {drot[2][0], drot[2][1], drot[2][2], 0.0f},
            {dx[0], dx[1], dx[2], 0.0f},
        };

        actor.transformation(actor.transformation() + trf);
        actor.updateVertexPositions();

        // Protect actors from escaping the world
        #pragma unroll
        for (unsigned int axis{0}; axis<3; ++axis) {
            actor.transformation()[3][0] = std::max(actor.transformation()[3][0], worldBoundaries.min()[0]);
            actor.transformation()[3][0] = std::min(actor.transformation()[3][0], worldBoundaries.max()[0]);
            actor.transformation()[3][1] = std::max(actor.transformation()[3][1], worldBoundaries.min()[1]);
            actor.transformation()[3][1] = std::min(actor.transformation()[3][1], worldBoundaries.max()[1]);
            actor.transformation()[3][2] = std::max(actor.transformation()[3][2], worldBoundaries.min()[2]);
            actor.transformation()[3][2] = std::min(actor.transformation()[3][2], worldBoundaries.max()[2]);
        }

        // Reset force and torque
        actor.force({0, 0, 0});
        actor.torque({0, 0, 0});
    }
}

// -----------------------------------------------------------------------------
void collideWorldSequential(std::vector<Actor>& actors, const Magnum::Range3D& worldBoundaries) {
    for (auto& actor : actors) {
        Wall collision{Wall::None};
        size_t collidingVertexIndex{std::numeric_limits<size_t>::max()};

        const auto& vertices{actor.vertexPositionsWorld()};
        const Magnum::Vector3& min{worldBoundaries.min()};
        const Magnum::Vector3& max{worldBoundaries.max()};
        Magnum::Vector3 normal{0.0f, 0.0f, 0.0f};

        for (size_t iVertex{0}; iVertex < vertices.size(); ++iVertex) {
            const Magnum::Vector3& vertex{vertices[iVertex]};
            if (vertex[0] <= min[0]) {
                collision=Wall::Xmin;
                collidingVertexIndex = iVertex;
                normal[0] = 1.0;
                break;
            }
            if (vertex[0] >= max[0]) {
                collision=Wall::Xmax;
                collidingVertexIndex = iVertex;
                normal[0] = -1.0;
                break;
            }
            if (vertex[1] <= min[1]) {
                collision=Wall::Ymin;
                collidingVertexIndex = iVertex;
                normal[1] = 1.0;
                break;
            }
            if (vertex[1] >= max[1]) {
                collision=Wall::Ymax;
                collidingVertexIndex = iVertex;
                normal[1] = -1.0;
                break;
            }
            if (vertex[2] <= min[2]) {
                collision=Wall::Zmin;
                collidingVertexIndex = iVertex;
                normal[2] = 1.0;
                break;
            }
            if (vertex[2] >= max[2]) {
                collision=Wall::Zmax;
                collidingVertexIndex = iVertex;
                normal[2] = -1.0;
                break;
            }
        }
        if (collision==Wall::None) {continue;}
        if (Magnum::Math::dot(actor.linearVelocity(), normal) > 0.0f) {
            continue;
        }

        const Magnum::Vector3& collidingVertexWorld{vertices[collidingVertexIndex]};

        const Magnum::Vector3 radius = collidingVertexWorld - actor.transformation().translation();
        const auto a = Magnum::Math::cross(radius, normal);
        const auto b = actor.inertiaInv() * a;
        const auto c = Magnum::Math::cross(b, radius);
        const auto d = Magnum::Math::dot(c, normal);
        const float impulse = (-1.0f - Constants::RestitutionCoefficient) *
                              Magnum::Math::dot(actor.linearVelocity(), normal) /
                              (1.0f/actor.mass() + d);

        Magnum::Vector3 addLinearV = (impulse / actor.mass()) * normal;
        Magnum::Vector3 addAngularV = impulse * b;

        actor.addVelocity(addLinearV, addAngularV);
        const float vy{actor.linearVelocity().y()};
        // TODO: implement better resting condition
        if (normal.y() > 0 && vy > 0 && vy < 0.01) {
            actor.addVelocity({0.0f, -1.0f*vy, 0.0f}, {0.0f, 0.0f, 0.0f});
        }
    }
}

// -----------------------------------------------------------------------------
void collideBroadSequential(std::vector<Actor>& actors, SequentialState& state) {
    // ===========================================
    // Sweep and prune algorithm as described in D.J. Tracy, S.R. Buss, B.M. Woods,
    // Efficient Large-Scale Sweep and Prune Methods with AABB Insertion and Removal, 2009
    // https://mathweb.ucsd.edu/~sbuss/ResearchWeb/EnhancedSweepPrune/SAP_paper_online.pdf
    // ===========================================
    auto cmp = [&actors](unsigned int axis, Edge edgeA, Edge edgeB){
        const float a{
            edgeA.isEnd ?
            actors[edgeA.actorIndex].axisAlignedBoundingBox().max()[axis] :
            actors[edgeA.actorIndex].axisAlignedBoundingBox().min()[axis]};
        const float b{
            edgeB.isEnd ?
            actors[edgeB.actorIndex].axisAlignedBoundingBox().max()[axis] :
            actors[edgeB.actorIndex].axisAlignedBoundingBox().min()[axis]};
        return a<b;
    };

    // Insertion sort
    for (unsigned int axis{0}; axis<3; ++axis) {
        auto& edges{state.sortedAABBEdges[axis]};
        for (size_t i{1}; i<2*Constants::NumActors; ++i) {
            for (size_t j{i}; j>0 && cmp(axis, edges[j], edges[j-1]); --j) {
                std::swap(edges[j], edges[j-1]);
            }
        }
    }

    // Second pass to determine overlaps
    std::unordered_set<uint16_t> current;
    std::array<SequentialState::OverlapSet, 3> overlaps; // for each axis
    for (unsigned int axis{0}; axis<3; ++axis) {
        auto& edges{state.sortedAABBEdges[axis]};
        for (size_t i{0}; i<2*Constants::NumActors; ++i) {
            Edge edge{edges[i]};
            if (edge.isEnd) {
                current.erase(edge.actorIndex);
                for (uint16_t otherActorIndex : current) {
                    overlaps[axis].insert(
                        edge.actorIndex < otherActorIndex ?
                        std::make_pair(edge.actorIndex, otherActorIndex) :
                        std::make_pair(otherActorIndex, edge.actorIndex)
                    );
                }
            } else {
                current.insert(edge.actorIndex);
            }
        }
    }

    // Find the intersection of overlaps across the 3 axes
    state.aabbOverlaps.clear();
    for (const std::pair<uint16_t,uint16_t> overlap : overlaps[0]) {
        if (overlaps[1].contains(overlap) && overlaps[2].contains(overlap)) {
            state.aabbOverlaps.insert(overlap);
        }
    }
    state.aabbOverlapsLastFrame = state.aabbOverlaps.size();
}

// -----------------------------------------------------------------------------
void collideNarrowSequential(std::vector<Actor>& actors, SequentialState& state) {
    // ===========================================
    // Algorithm finding the closest vertex-triangle pair from two bodies
    // and comparing against a fixed threshold
    // ===========================================
    for (const auto [iActorA, iActorB] : state.aabbOverlaps) {
        const auto& verticesA{actors[iActorA].vertexPositionsWorld()};
        const auto& verticesB{actors[iActorB].vertexPositionsWorld()};
        const auto& indicesA{actors[iActorA].meshData().indicesAsArray()};
        const auto& indicesB{actors[iActorB].meshData().indicesAsArray()};
        size_t nTrianglesA{indicesA.size()/3};
        size_t nTrianglesB{indicesB.size()/3};
        sycl::float3 bestVertex{0.0f, 0.0f, 0.0f};
        sycl::float3 bestTrianglePoint{0.0f, 0.0f, 0.0f};
        sycl::float3 bestTriangleNorm{0.0f, 0.0f, 0.0f};
        bool bestTriangleFromA{false};
        float smallestDistanceSquared{std::numeric_limits<float>::max()};
        for (size_t j{0}; j<nTrianglesB; ++j) {
            sycl::float3 A{Util::toSycl(verticesB[indicesB[3*j]])};
            sycl::float3 B{Util::toSycl(verticesB[indicesB[3*j+1]])};
            sycl::float3 C{Util::toSycl(verticesB[indicesB[3*j+2]])};

            // Skip degenerate triangles
            if (Util::equal(A,B) || Util::equal(B,C)) {continue;}

            std::array<sycl::float3,3> triangle{A, B, C};
            const auto closest = Util::closestPointOnTriangleND(triangle, verticesA);
            if (closest.distanceSquared < smallestDistanceSquared) {
                smallestDistanceSquared = closest.distanceSquared;
                bestVertex = sycl::float3{verticesA[0][closest.iVertex], verticesA[1][closest.iVertex], verticesA[2][closest.iVertex]};
                bestTrianglePoint = closest.bestPointOnTriangle;
                bestTriangleNorm = sycl::cross(B-A, C-A);
                bestTriangleNorm /= sycl::length(bestTriangleNorm);
                if (sycl::dot(bestTriangleNorm, bestTrianglePoint - Util::toSycl(actors[iActorB].transformation_const().translation())) < 0) {
                    bestTriangleNorm *= -1.0f;
                }
            }
        }
        for (size_t j{0}; j<nTrianglesA; ++j) {
            sycl::float3 A{Util::toSycl(verticesA[indicesA[3*j]])};
            sycl::float3 B{Util::toSycl(verticesA[indicesA[3*j+1]])};
            sycl::float3 C{Util::toSycl(verticesA[indicesA[3*j+2]])};

            // Skip degenerate triangles
            if (Util::equal(A,B) || Util::equal(B,C)) {continue;}

            std::array<sycl::float3,3> triangle{A, B, C};
            const auto closest = Util::closestPointOnTriangleND(triangle, verticesB);
            if (closest.distanceSquared < smallestDistanceSquared) {
                smallestDistanceSquared = closest.distanceSquared;
                bestVertex = sycl::float3{verticesB[0][closest.iVertex], verticesB[1][closest.iVertex], verticesB[2][closest.iVertex]};
                bestTrianglePoint = closest.bestPointOnTriangle;
                bestTriangleFromA = true;
                bestTriangleNorm = sycl::cross(B-A, C-A);
                bestTriangleNorm /= sycl::length(bestTriangleNorm);
                if (sycl::dot(bestTriangleNorm, bestTrianglePoint - Util::toSycl(actors[iActorA].transformation_const().translation())) < 0) {
                    bestTriangleNorm *= -1.0f;
                }
            }
        }
        if (smallestDistanceSquared < Constants::NarrowPhaseCollisionThreshold) {
            Magnum::Vector3 collisionPoint = Util::toMagnum(0.5f*(bestVertex+bestTrianglePoint));
            if (bestTriangleFromA) {
                detail::impulseCollision(actors[iActorA], actors[iActorB], collisionPoint, Util::toMagnum(bestTriangleNorm));
            } else {
                detail::impulseCollision(actors[iActorB], actors[iActorA], collisionPoint, Util::toMagnum(bestTriangleNorm));
            }
            // Move the two actors away to avoid clipping and triggering
            // the collision multiple times
            float smallestDistance = sycl::sqrt(smallestDistanceSquared);
            const Magnum::Vector3 shiftA = smallestDistance * actors[iActorA].linearVelocity().normalized();
            const Magnum::Vector3 shiftB = smallestDistance * actors[iActorB].linearVelocity().normalized();
            actors[iActorA].transformation(actors[iActorA].transformation_const() + Magnum::Matrix4{
                Magnum::Vector4{0.0f},
                Magnum::Vector4{0.0f},
                Magnum::Vector4{0.0f},
                {shiftA[0], shiftA[1], shiftA[2], 0.0f}});
            actors[iActorB].transformation(actors[iActorB].transformation_const() + Magnum::Matrix4{
                Magnum::Vector4{0.0f},
                Magnum::Vector4{0.0f},
                Magnum::Vector4{0.0f},
                {shiftB[0], shiftB[1], shiftB[2], 0.0f}});
            actors[iActorA].updateVertexPositions();
            actors[iActorB].updateVertexPositions();
        }
    }
}

// -----------------------------------------------------------------------------
void impulseCollision(Actor& a, Actor& b, const Magnum::Vector3& point, const Magnum::Vector3& normal) {
    // ===========================================
    // See "Impulse-based reaction model" in
    // https://en.wikipedia.org/wiki/Collision_response
    // ===========================================
    using Magnum::Math::cross;
    using Magnum::Math::dot;
    Magnum::Vector3 ra = point - a.transformation_const().translation();
    Magnum::Vector3 rb = point - b.transformation_const().translation();
    Magnum::Vector3 vpa = a.linearVelocity() + cross(a.angularVelocity(), ra);
    Magnum::Vector3 vpb = b.linearVelocity() + cross(b.angularVelocity(), rb);
    Magnum::Vector3 vr = vpb - vpa;
    Magnum::Vector3 ta = a.inertiaInv()*cross(ra,normal);
    Magnum::Vector3 tb = b.inertiaInv()*cross(rb,normal);
    float impulse =
        (-1.0f - Constants::RestitutionCoefficient) *
        dot(vr,normal) / (
            1.0f/a.mass() +
            1.0f/b.mass() +
            dot(
                 cross(ta, ra) +
                 cross(tb, rb)
                , normal
            )
        );
    Magnum::Vector3 addLinVA = -1.0f * normal * impulse / a.mass();
    Magnum::Vector3 addLinVB = normal * impulse / b.mass();
    Magnum::Vector3 addAngVA = -1.0f * impulse * ta;
    Magnum::Vector3 addAngVB = impulse * tb;
    a.addVelocity(addLinVA, addAngVA);
    b.addVelocity(addLinVB, addAngVB);
}
} // namespace detail

// -----------------------------------------------------------------------------
void simulateSequential(float dtime, std::vector<Actor>& actors, SequentialState& state) {
    for (Actor& actor : actors) {
        // Fix floating point loss of orthogonality in the rotation matrix
        Util::orthonormaliseRotation(actor.transformation());
    }
    detail::simulateMotionSequential(dtime, actors, state.worldBoundaries);
    detail::collideWorldSequential(actors, state.worldBoundaries);
    detail::collideBroadSequential(actors, state);
    detail::collideNarrowSequential(actors, state);
}

} // namespace CollisionSim::Simulation
