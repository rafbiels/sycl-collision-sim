/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "Simulation.h"
#include "Constants.h"
#include "Util.h"
#include "Wall.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/user_defined_reductions.hpp>
#include <sycl/ext/oneapi/experimental/group_helpers_sorters.hpp>
#include <limits>
#include <numeric>
#include <unordered_set>

namespace CollisionSim::Simulation {

// -----------------------------------------------------------------------------
void simulateMotionSequential(float dtime, std::vector<Actor>& actors) {
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
        if (normal.y() > 0 && vy > 0 && vy < 0.1) {
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
    std::array<Util::OverlapSet, 3> overlaps; // for each axis
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
                impulseCollision(actors[iActorA], actors[iActorB], collisionPoint, Util::toMagnum(bestTriangleNorm));
            } else {
                impulseCollision(actors[iActorB], actors[iActorA], collisionPoint, Util::toMagnum(bestTriangleNorm));
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
    // std::cout <<    "ra \t= ("    << ra[0] << ',' << ra[1] << ',' << ra[2]
    //           << ")\nrb \t= ("  << rb[0] << ',' << rb[1] << ',' << rb[2]
    //           << ")\nnormal \t= ("  << normal[0] << ',' << normal[1] << ',' << normal[2]
    //           << ")\nvpa \t= (" << vpa[0] << ',' << vpa[1] << ',' << vpa[2]
    //           << ")\nvpb \t= (" << vpb[0] << ',' << vpb[1] << ',' << vpb[2]
    //           << ")\nvr \t= ("  << vr[0] << ',' << vr[1] << ',' << vr[2]
    //           << ")\nta \t= ("  << ta[0] << ',' << ta[1] << ',' << ta[2]
    //           << ")\ntb \t= ("  << tb[0] << ',' << tb[1] << ',' << tb[2]
    //           << ")\naddLinVA \t= ("  << addLinVA[0] << ',' << addLinVA[1] << ',' << addLinVA[2]
    //           << ")\naddLinVB \t= ("  << addLinVB[0] << ',' << addLinVB[1] << ',' << addLinVB[2]
    //           << ")\naddAngVA \t= ("  << addAngVA[0] << ',' << addAngVA[1] << ',' << addAngVA[2]
    //           << ")\naddAngVB \t= ("  << addAngVB[0] << ',' << addAngVB[1] << ',' << addAngVB[2]
    //           << ")\nimpulse \t= " << impulse
    //           << "\n\n";
    a.addVelocity(addLinVA, addAngVA);
    b.addVelocity(addLinVB, addAngVB);
}

// -----------------------------------------------------------------------------
void simulateSequential(float dtime, std::vector<Actor>& actors, SequentialState& state) {
    for (Actor& actor : actors) {
        // Fix floating point loss of orthogonality in the rotation matrix
        Util::orthonormaliseRotation(actor.transformation());
    }
    simulateMotionSequential(dtime, actors);
    collideWorldSequential(actors, state.worldBoundaries);
    collideBroadSequential(actors, state);
    collideNarrowSequential(actors, state);
}

// -----------------------------------------------------------------------------
void simulateParallel(float dtime, std::vector<Actor>& actors, ParallelState& state, sycl::queue& queue) {
    using float3x3 = CollisionSim::ParallelState::float3x3;
    // Copy inputs from Actor objects to serial state data
    for (size_t iActor{0}; iActor<Constants::NumActors; ++iActor) {
        state.linearVelocity.hostContainer[iActor] = Util::toSycl(actors[iActor].linearVelocity());
        state.angularVelocity.hostContainer[iActor] = Util::toSycl(actors[iActor].angularVelocity());
        state.force.hostContainer[iActor] = Util::toSycl(actors[iActor].force());
        state.torque.hostContainer[iActor] = Util::toSycl(actors[iActor].torque());
    }
    try {
        std::vector<sycl::event> h2dCopyEvents{
            state.linearVelocity.copyToDevice(),
            state.angularVelocity.copyToDevice(),
            state.force.copyToDevice(),
            state.torque.copyToDevice()
        };

        // Device pointers to be captured by lambda and copied to device
        // - this is to avoid dereferencing on device the state host pointer
        float* worldBoundaries = state.worldBoundaries.devicePointer;
        float* mass = state.mass.devicePointer;
        uint16_t* actorIndices = state.actorIndices.devicePointer;
        sycl::float3* linearVelocity = state.linearVelocity.devicePointer;
        float3x3* inertiaInv = state.inertiaInv.devicePointer;
        sycl::float3* translation = state.translation.devicePointer;
        sycl::float3* addLinearVelocity = state.addLinearVelocity.devicePointer;
        sycl::float3* addAngularVelocity = state.addAngularVelocity.devicePointer;
        Wall* wallCollisions = state.wallCollisions.devicePointer;
        float3x3* bodyInertiaInv = state.bodyInertiaInv.devicePointer;
        uint16_t* numVertices = state.numVertices.devicePointer;
        uint32_t* verticesOffset = state.verticesOffset.devicePointer;
        sycl::float3* bodyVertices = state.bodyVertices.devicePointer;
        sycl::float3* worldVertices = state.worldVertices.devicePointer;
        uint16_t* numTriangles = state.numTriangles.devicePointer;
        uint32_t* trianglesOffset = state.trianglesOffset.devicePointer;
        sycl::uint3* triangles = state.triangles.devicePointer;
        sycl::float3* angularVelocity = state.angularVelocity.devicePointer;
        float3x3* rotation = state.rotation.devicePointer;
        sycl::float3* force = state.force.devicePointer;
        sycl::float3* torque = state.torque.devicePointer;
        std::array<sycl::float2*,3> aabb {
            state.aabb[0].devicePointer,
            state.aabb[1].devicePointer,
            state.aabb[2].devicePointer
        };
        std::array<Edge*,3> sortedAABBEdges {
            state.sortedAABBEdges[0].devicePointer,
            state.sortedAABBEdges[1].devicePointer,
            state.sortedAABBEdges[2].devicePointer
        };
        bool* aabbOverlaps = state.aabbOverlaps.devicePointer;

        sycl::event actorKernelEvent = queue.submit([&](sycl::handler& cgh){
            cgh.depends_on(h2dCopyEvents);
            cgh.parallel_for<class actor_kernel>(Constants::NumActors, [=](sycl::id<1> id){
                // Compute linear and angular momentum
                auto linearMomentum = mass[id] * linearVelocity[id];
                auto angularMomentum = Util::mvmul(Util::inverse(inertiaInv[id]), angularVelocity[id]);

                linearMomentum += force[id] * dtime;
                angularMomentum += torque[id] * dtime;

                // Compute linear and angular velocity
                linearVelocity[id] = linearMomentum / mass[id];
                inertiaInv[id] = Util::mmul(
                    Util::mmul(rotation[id], bodyInertiaInv[id]),
                    Util::transpose(rotation[id])); // R * Ib^-1 * R^T
                angularVelocity[id] = Util::mvmul(inertiaInv[id], angularMomentum);

                // Apply translation
                translation[id] += linearVelocity[id] * dtime;

                // Apply rotation
                auto star = [](const sycl::float3& v) constexpr {
                    return std::array<sycl::float3,3>{
                        sycl::float3{ 0.0f,  v[2], -v[1]},
                        sycl::float3{-v[2],  0.0f,  v[0]},
                        sycl::float3{ v[1], -v[0],  0.0f}
                    };
                };
                std::array<sycl::float3,3> drot = Util::msmul(
                    Util::mmul(star(angularVelocity[id]), rotation[id]),
                    dtime);
                rotation[id][0] += drot[0];
                rotation[id][1] += drot[1];
                rotation[id][2] += drot[2];
            });
        });

        // Update vertex positions and calculate world collisions
        sycl::event vertexKernelEvent = queue.submit([&](sycl::handler& cgh){
            cgh.depends_on(actorKernelEvent);
            cgh.parallel_for<class vertex_kernel>(state.numAllVertices, [=](sycl::id<1> id){
                uint16_t iActor = actorIndices[id];

                worldVertices[id] = sycl::float3{
                    // x
                    rotation[iActor][0][0]*bodyVertices[id][0] +
                    rotation[iActor][1][0]*bodyVertices[id][1] +
                    rotation[iActor][2][0]*bodyVertices[id][2] +
                    translation[iActor][0],
                    // y
                    rotation[iActor][0][1]*bodyVertices[id][0] +
                    rotation[iActor][1][1]*bodyVertices[id][1] +
                    rotation[iActor][2][1]*bodyVertices[id][2] +
                    translation[iActor][1],
                    // z
                    rotation[iActor][0][2]*bodyVertices[id][0] +
                    rotation[iActor][1][2]*bodyVertices[id][1] +
                    rotation[iActor][2][2]*bodyVertices[id][2] +
                    translation[iActor][2]
                };

                Wall collision{Wall::None};
                collision |= (static_cast<WallUnderlyingType>(worldVertices[id][0] <= worldBoundaries[0]) << 0);
                collision |= (static_cast<WallUnderlyingType>(worldVertices[id][0] >= worldBoundaries[1]) << 1);
                collision |= (static_cast<WallUnderlyingType>(worldVertices[id][1] <= worldBoundaries[2]) << 2);
                collision |= (static_cast<WallUnderlyingType>(worldVertices[id][1] >= worldBoundaries[3]) << 3);
                collision |= (static_cast<WallUnderlyingType>(worldVertices[id][2] <= worldBoundaries[4]) << 4);
                collision |= (static_cast<WallUnderlyingType>(worldVertices[id][2] >= worldBoundaries[5]) << 5);

                sycl::float3 normal = wallNormal(collision);
                sycl::float3 radius{worldVertices[id] - translation[iActor]};
                sycl::float3 a{sycl::cross(radius, normal)};
                sycl::float3 b{Util::mvmul(inertiaInv[iActor], a)};
                sycl::float3 c{sycl::cross(b, radius)};
                float d{sycl::dot(c, normal)};
                float impulse = (-1.0f - Constants::RestitutionCoefficient) *
                                sycl::dot(linearVelocity[iActor], normal) /
                                (1.0f/mass[iActor] + d);

                addLinearVelocity[id] = (impulse / mass[iActor]) * normal;
                addAngularVelocity[id] = impulse * b;
                bool ignoreAwayFromWall{sycl::dot(linearVelocity[iActor], normal) > 0.0f};
                wallCollisions[id] = static_cast<Wall>(
                    static_cast<WallUnderlyingType>(collision) *
                    static_cast<WallUnderlyingType>(!ignoreAwayFromWall));
            });
        });

        // Calculate the axis-align bounding boxes for each actor
        // The use of sycl::joint_reduce requires 1D arrays of vx, vy, vz
        // so we copy the AoS global memory vertices into local memory SoA
        sycl::event aabbKernelEvent = queue.submit([&](sycl::handler& cgh){
            cgh.depends_on(vertexKernelEvent);
            constexpr static size_t aabbWorkGroupSize{32};
            const sycl::nd_range<1> aabbRange{Constants::NumActors*aabbWorkGroupSize,aabbWorkGroupSize};
            std::array localVertices{
                sycl::local_accessor<float,1>{sycl::range<1>{state.maxNumVerticesPerActor},cgh},
                sycl::local_accessor<float,1>{sycl::range<1>{state.maxNumVerticesPerActor},cgh},
                sycl::local_accessor<float,1>{sycl::range<1>{state.maxNumVerticesPerActor},cgh}};
            cgh.parallel_for<class aabb_kernel>(aabbRange, [=](sycl::nd_item<1> it){
                size_t iActor = it.get_group_linear_id();

                sycl::float3* actorVertices = worldVertices + verticesOffset[iActor];
                size_t numVerticesPerThread{1 + numVertices[iActor]/aabbWorkGroupSize};
                for (size_t i{0}; i<numVerticesPerThread; ++i) {
                    size_t iVertex{it.get_local_linear_id() * numVerticesPerThread + i};
                    if (iVertex<numVertices[iActor]) {
                        localVertices[0][iVertex] = actorVertices[iVertex][0];
                        localVertices[1][iVertex] = actorVertices[iVertex][1];
                        localVertices[2][iVertex] = actorVertices[iVertex][2];
                    }
                }

                sycl::group_barrier(it.get_group());

                #pragma unroll
                for (unsigned int axis{0}; axis<3; ++axis) {
                    aabb[axis][iActor] = sycl::float2{
                        sycl::joint_reduce(
                            it.get_group(),
                            localVertices[axis].begin(),
                            localVertices[axis].begin()+numVertices[iActor],
                            sycl::minimum{}),
                        sycl::joint_reduce(
                            it.get_group(),
                            localVertices[axis].begin(),
                            localVertices[axis].begin()+numVertices[iActor],
                            sycl::maximum{})
                    };
                }
            });
        });

        // Sort the AABB edges using odd-even merge-sort
        // Given the small size of the problem we can avoid submitting N(=2*NumActors) kernels.
        // Instead, we can submit one kernel with a single work-group and exploit a work-group barrier.
        // This requires that N is smaller than the maximum work-group size of the GPU we're using.
        sycl::event aabbSortKernelEvent = queue.submit([&](sycl::handler& cgh){
            cgh.depends_on(aabbKernelEvent);
            cgh.parallel_for<class aabb_sort_kernel>(sycl::nd_range<1>{Constants::NumActors, Constants::NumActors}, [=](sycl::nd_item<1> item){
                auto edgeValue = [aabb](unsigned int axis, Edge e) constexpr -> float {
                    return e.isEnd ? aabb[axis][e.actorIndex][1] : aabb[axis][e.actorIndex][0];
                };
                auto edgeGreater = [aabb, &edgeValue](unsigned int axis, Edge edgeA, Edge edgeB) constexpr -> bool {
                    return edgeValue(axis, edgeA) > edgeValue(axis, edgeB);
                };
                auto compareExchange = [aabb, &edgeGreater](unsigned int axis, Edge& edgeA, Edge& edgeB) constexpr -> void {
                    if (edgeGreater(axis,edgeA,edgeB)) {std::swap(edgeA, edgeB);}
                };
                size_t id = item.get_global_linear_id();
                for (size_t step{0}; step < 2*Constants::NumActors; ++step) {
                    size_t i = id * 2 + step%2;
                    if ((i+1) >= (2*Constants::NumActors)) {return;}
                    compareExchange(0, sortedAABBEdges[0][i], sortedAABBEdges[0][i+1]);
                    compareExchange(1, sortedAABBEdges[1][i], sortedAABBEdges[1][i+1]);
                    compareExchange(2, sortedAABBEdges[2][i], sortedAABBEdges[2][i+1]);
                    sycl::group_barrier(item.get_group());
                }
            });
        });

        // Start copying back to host data needed there which won't change beyond this point
        std::vector<sycl::event> d2hCopyEvents{
            state.translation.copyToHost(actorKernelEvent),
            state.rotation.copyToHost(actorKernelEvent),
            state.wallCollisions.copyToHost(vertexKernelEvent),
            state.addLinearVelocity.copyToHost(vertexKernelEvent),
            state.addAngularVelocity.copyToHost(vertexKernelEvent),
        };

        // Find overlapping AABB pairs
        sycl::event aabbOverlapKernelEvent = queue.submit([&](sycl::handler& cgh){
            cgh.depends_on(aabbSortKernelEvent);
            cgh.parallel_for<class aabb_overlap_kernel>(Constants::NumActorPairs, [=](sycl::id<1> id){
                size_t iActorA{Constants::ActorPairs[id].first};
                size_t iActorB{Constants::ActorPairs[id].second};
                sycl::int3 posStartA{-1};
                sycl::int3 posStartB{-1};
                sycl::int3 posEndA{-1};
                sycl::int3 posEndB{-1};
                for (int iEdge{0}; iEdge<static_cast<int>(2*Constants::NumActors); ++iEdge) {
                    #pragma unroll
                    for (int axis{0}; axis<3; ++axis) {
                        const Edge& edge{sortedAABBEdges[axis][iEdge]};
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
                aabbOverlaps[id] = (
                    overlap(posStartA[0], posEndA[0], posStartB[0], posEndB[0]) &&
                    overlap(posStartA[1], posEndA[1], posStartB[1], posEndB[1]) &&
                    overlap(posStartA[2], posEndA[2], posStartB[2], posEndB[2])
                );
            });
        });

        // Copy back overlaps to host in order to find out how many narrow-phase kernels to submit
        state.aabbOverlaps.copyToHost(aabbOverlapKernelEvent).wait_and_throw();
        std::unordered_set<uint16_t> overlappingActors;
        USMData<int,Constants::NumActors> usmPairedActorIndices{queue,{-1}};
        size_t numTrianglesToCheck{0};
        size_t numActorsToCheck{0};
        for (size_t iPair{0}; iPair<Constants::NumActorPairs; ++iPair) {
            if (state.aabbOverlaps.hostContainer[iPair]) {
                const std::pair<size_t,size_t>& pair{Constants::ActorPairs[iPair]};
                usmPairedActorIndices.hostContainer[pair.first] = pair.second;
                usmPairedActorIndices.hostContainer[pair.second] = pair.first;
                // Corrade::Utility::Debug{} << "Actors " << pair.first << " and " << pair.second << " overlap AABB";
                for (size_t iActor : {pair.first, pair.second}) {
                    if (overlappingActors.insert(iActor).second) {
                        numTrianglesToCheck += state.numTriangles.hostContainer[iActor];
                        ++numActorsToCheck;
                    };
                }
            }
        }
        if (!overlappingActors.empty()) {
            // Corrade::Utility::Debug{} << "AABB overlapping actors: " << overlappingActors
            //     << " with " << numTrianglesToCheck << " triangles to check";
            USMData<uint16_t> usmOverlappingActors{queue, overlappingActors.size()};
            usmOverlappingActors.hostContainer.assign(overlappingActors.begin(), overlappingActors.end());
            sycl::event copyOverlappingActorsEvent = usmOverlappingActors.copyToDevice();
            sycl::event copyPairedActorIndicesEvent = usmPairedActorIndices.copyToDevice();
            uint16_t* dptrOverlappingActors = usmOverlappingActors.devicePointer;
            int* dptrPairedActorIndices = usmPairedActorIndices.devicePointer;

            // Allocate USM container for the output of the triangle-vertex matching
            // float4 = {float3 point on triangle, float3 normal, float distance squared}
            struct TVMatch {
                sycl::float3 pointOnTriangle{0.0f};
                sycl::float3 normal{0.0f};
                float dsq{std::numeric_limits<float>::max()};
            };
            USMData<TVMatch> usmTriangleVertexMatch{queue, numActorsToCheck*Constants::MaxNumTriangles};
            TVMatch* dptrTriangleVertexMatch = usmTriangleVertexMatch.devicePointer;
            // Reset the triangle-vertex match data on the device
            sycl::event resetTriangleVertexMatchEvent = queue.submit([&](sycl::handler& cgh){
                cgh.parallel_for<class tv_reset_kernel>(usmTriangleVertexMatch.size(), [=](sycl::id<1> id){
                    dptrTriangleVertexMatch[id] = TVMatch{};
                });
            });

            sycl::event triangleTransformKernelEvent = queue.submit([&](sycl::handler& cgh){
                cgh.depends_on({copyOverlappingActorsEvent,copyPairedActorIndicesEvent,resetTriangleVertexMatchEvent});
                // auto os = sycl::stream{1024,65535,cgh};
                cgh.parallel_for<class narrow_phase_kernel>(numTrianglesToCheck, [=](sycl::id<1> id){
                    unsigned int iTriangle{0};
                    unsigned int iActor{0};
                    unsigned int iActorTriangle{0};
                    unsigned int iOverlap{0};
                    unsigned int offset{0};
                    for (unsigned int iThisOverlap{0}; iThisOverlap<numActorsToCheck; ++iThisOverlap) {
                        const unsigned int iThisActor = dptrOverlappingActors[iThisOverlap];
                        const uint16_t numTrianglesThisActor = numTriangles[iThisActor];
                        const unsigned int iTriangleThisActor{static_cast<unsigned int>(id)-offset};
                        if (iTriangleThisActor < numTrianglesThisActor) {
                            iTriangle = trianglesOffset[iThisActor] + iTriangleThisActor;
                            iActor = iThisActor;
                            iActorTriangle = iTriangleThisActor;
                            iOverlap = iThisOverlap;
                        }
                        offset += numTrianglesThisActor;
                    }
                    const unsigned int vertexOffset = verticesOffset[iActor];
                    const sycl::uint3& triangleIndices{triangles[iTriangle]};
                    std::array<sycl::float3,3> triangle{
                        worldVertices[vertexOffset+triangleIndices[0]],
                        worldVertices[vertexOffset+triangleIndices[1]],
                        worldVertices[vertexOffset+triangleIndices[2]]
                    };
                    auto transformed = Util::triangleTransform(triangle);
                    const auto& rot = transformed[1];
                    const auto& negRot = transformed[2];

                    unsigned int iOtherActor{static_cast<unsigned int>(dptrPairedActorIndices[iActor])};
                    const unsigned int vertexOffsetOtherActor = verticesOffset[iOtherActor];
                    const unsigned int numVerticesOtherActor = numVertices[iOtherActor];

                    float smallestDistanceSquared{std::numeric_limits<float>::max()};
                    sycl::float3 bestPointOnTriangle{0.0f, 0.0f, 0.0f};
                    unsigned int bestVertexIndex{std::numeric_limits<unsigned int>::max()};

                    for (unsigned int iVertex{0}; iVertex<numVerticesOtherActor; ++iVertex) {
                        sycl::float3 P{worldVertices[vertexOffsetOtherActor+iVertex]};
                        P -= triangle[0];
                        P = Util::mvmul(rot,P);

                        const auto [closestPoint, distanceSquared] = Util::closestPointOnTriangle(transformed[0], P);

                        if (distanceSquared < smallestDistanceSquared) {
                            smallestDistanceSquared = distanceSquared;
                            bestPointOnTriangle = closestPoint;
                            bestVertexIndex = iVertex;
                        }
                    }
                    bestPointOnTriangle = Util::mvmul(negRot, bestPointOnTriangle) + triangle[0];
                    sycl::float3 normal = sycl::cross(triangle[1]-triangle[0], triangle[2]-triangle[0]);
                    // bestTriangleNorm /= sycl::length(bestTriangleNorm);
                    // if (sycl::dot(bestTriangleNorm, bestTrianglePoint - Util::toSycl(actors[iActorA].transformation_const().translation())) < 0) {
                    //     bestTriangleNorm *= -1.0f;
                    // }
                    normal /= sycl::length(normal);
                    sycl::float3 radius{bestPointOnTriangle - translation[iActor]};
                    float direction = (sycl::dot(normal, radius) < 0) ? -1.0f : 1.0f;
                    normal *= direction;

                    // os << "id " << id << " setting TVMatch for triangle id " << iOverlap*Constants::MaxNumTriangles + iActorTriangle
                    //    << " to (" << bestPointOnTriangle[0] << ", " << bestPointOnTriangle[1] << ", " << bestPointOnTriangle[2] << ") "
                    //    << smallestDistanceSquared << "\n";
                    dptrTriangleVertexMatch[iOverlap*Constants::MaxNumTriangles + iActorTriangle] = {
                        sycl::float3{bestPointOnTriangle[0], bestPointOnTriangle[1], bestPointOnTriangle[2]},
                        normal,
                        smallestDistanceSquared
                    };
                });
            });

            // Allocate USM container for the output of the closest-distance triangle-vertex matching
            USMData<TVMatch> usmTriangleBestMatch{queue, numActorsToCheck};
            TVMatch* dptrTriangleBestMatch = usmTriangleBestMatch.devicePointer;
            // constexpr static size_t wgSize{256};
            const sycl::nd_range<1> reduceRange{numActorsToCheck*Constants::MaxNumTriangles, Constants::MaxNumTriangles};

            struct MyReduce {
                TVMatch operator()(const TVMatch& a, const TVMatch& b) const {
                    return a.dsq<b.dsq ? a : b;
                }
            };

            // For each actor out of numActorsToCheck, reduce all triangle-vertex pairs to the one with the smallest distance
            sycl::event triangleVertexReduceKernelEvent = queue.submit([&](sycl::handler& cgh){
                cgh.depends_on(triangleTransformKernelEvent);
                size_t temp_memory_size = Constants::MaxNumTriangles*sizeof(TVMatch);
                sycl::local_accessor<std::byte, 1> scratch{temp_memory_size, cgh};
                // auto os = sycl::stream{1024,65535,cgh};
                cgh.parallel_for<class tv_reduce_kernel>(reduceRange, [=](sycl::nd_item<1> item){
                    const auto groupId{item.get_group_linear_id()};
                    const auto localId{item.get_local_linear_id()};
                    TVMatch* start = dptrTriangleVertexMatch + groupId*Constants::MaxNumTriangles;
                    TVMatch* end = start + Constants::MaxNumTriangles;
                    sycl::ext::oneapi::experimental::group_with_scratchpad handle{
                        item.get_group(), sycl::span{&scratch[0], temp_memory_size}};
                    dptrTriangleBestMatch[groupId] =
                        sycl::ext::oneapi::experimental::joint_reduce(handle, start, end, MyReduce{});
                });
            });

            // FIXME: Force synchronisation after triangleVertexReduceKernelEvent for now, which will be
            // replaced with real dependency on d2h copy of the final narrow-phase collision output
            sycl::event triangleVertexReduceCopyEvent = usmTriangleBestMatch.copyToHost(triangleVertexReduceKernelEvent);
            d2hCopyEvents.push_back(triangleVertexReduceCopyEvent);

            // DEBUGGING ONLY
            triangleVertexReduceCopyEvent.wait_and_throw();
            usmTriangleVertexMatch.copyToHost(triangleTransformKernelEvent).wait_and_throw();
            // Corrade::Utility::Debug{} << "TriangleBestMatch size: " << usmTriangleBestMatch.hostContainer.size();
            // for (const auto& tv : usmTriangleBestMatch.hostContainer) {
            //     Corrade::Utility::Debug{} << Magnum::Vector3{tv.x,tv.y,tv.z} << tv.dsq;
            // }
            // Corrade::Utility::Debug{} << "TriangleVertexMatch size: " << usmTriangleVertexMatch.hostContainer.size();
            // for (const auto& tv : usmTriangleVertexMatch.hostContainer) {
            //     Corrade::Utility::Debug{} << Magnum::Vector3{tv.x,tv.y,tv.z} << tv.dsq;
            // }
            size_t iOverlap{0};
            std::vector<std::pair<uint16_t,uint16_t>> collidingPairs;
            std::vector<TVMatch> collidingTV;
            std::vector<char> triangleFromFirstInPair;
            for (uint16_t iActor : overlappingActors) {
                const TVMatch& tv = usmTriangleBestMatch.hostContainer[iOverlap];
                if (tv.dsq >= Constants::NarrowPhaseCollisionThreshold) {
                    ++iOverlap;
                    continue;
                }

                uint16_t iOtherActor = usmPairedActorIndices.hostContainer[iActor];
                std::pair<uint16_t,uint16_t> pair = iActor < iOtherActor ?
                    std::pair<uint16_t,uint16_t>{iActor,iOtherActor} :
                    std::pair<uint16_t,uint16_t>{iOtherActor,iActor};
                const auto it = std::find(collidingPairs.begin(), collidingPairs.end(), pair);
                if (it == collidingPairs.end()) {
                    collidingPairs.push_back(pair);
                    collidingTV.push_back(tv);
                    triangleFromFirstInPair.push_back(pair.first==iActor);
                } else {
                    size_t index = std::distance(collidingPairs.begin(),it);
                    if (collidingPairs[index]!=pair) {throw std::runtime_error{"wrong index"};}
                    if (tv.dsq < collidingTV[index].dsq) {
                        collidingTV[index] = tv;
                        triangleFromFirstInPair[index] = (pair.first==iActor);
                    }
                }
                ++iOverlap;
            }
            // for (size_t iPair{0}; iPair<collidingPairs.size(); ++iPair) {
            //     const auto& pair = collidingPairs[iPair];
            //     const auto& tv = collidingTV[iPair];
            //     Corrade::Utility::Debug{} << "Collision (" << pair.first << "," << pair.second << "): "
            //         << Magnum::Vector3{tv.pointOnTriangle[0],tv.pointOnTriangle[1],tv.pointOnTriangle[2]} << tv.dsq;
            // }

            // Copy the fully-reduced collision data to the device
            USMData<std::pair<uint16_t,uint16_t>> usmCollidingPairs{queue, collidingPairs};
            USMData<TVMatch> usmCollidingTV{queue, collidingTV};
            USMData<char> usmTriangleFromFirstInPair{queue, triangleFromFirstInPair};
            sycl::event h2dCollidingPairsEvent = usmCollidingPairs.copyToDevice();
            sycl::event h2dCollidingTVEvent = usmCollidingTV.copyToDevice();
            sycl::event h2dTriangleFromFirstInPair = usmTriangleFromFirstInPair.copyToDevice();
            std::pair<uint16_t,uint16_t>* dptrCollidingPairs = usmCollidingPairs.devicePointer;
            TVMatch* dptrCollidingTV = usmCollidingTV.devicePointer;
            char* dptrTriangleFromFirstInPair = usmTriangleFromFirstInPair.devicePointer;
            // Submit the impulse collision kernel
            sycl::event impulseCollisionKernelEvent = queue.submit([&](sycl::handler& cgh){
                cgh.depends_on({h2dCollidingPairsEvent, h2dCollidingTVEvent});
                // auto os = sycl::stream{1024,65535,cgh};
                cgh.parallel_for<class impulse_collision_kernel>(collidingPairs.size(), [=](sycl::id<1> id){
                    uint16_t iActorA{0}, iActorB{0};
                    if (dptrTriangleFromFirstInPair[id]) {
                        iActorA = dptrCollidingPairs[id].first;
                        iActorB = dptrCollidingPairs[id].second;
                    } else {
                        iActorA = dptrCollidingPairs[id].second;
                        iActorB = dptrCollidingPairs[id].first;
                    }
                    const sycl::float3& point{dptrCollidingTV[id].pointOnTriangle};
                    const sycl::float3& normal{dptrCollidingTV[id].normal};
                    sycl::float3 ra = point - translation[iActorA];
                    sycl::float3 rb = point - translation[iActorB];
                    sycl::float3 vpa = linearVelocity[iActorA] + sycl::cross(angularVelocity[iActorA], ra);
                    sycl::float3 vpb = linearVelocity[iActorB] + sycl::cross(angularVelocity[iActorB], rb);
                    sycl::float3 vr = vpb - vpa;
                    sycl::float3 ta = Util::mvmul(inertiaInv[iActorA], sycl::cross(ra,normal));
                    sycl::float3 tb = Util::mvmul(inertiaInv[iActorB], sycl::cross(rb,normal));
                    float impulse =
                        (-1.0f - Constants::RestitutionCoefficient) *
                        sycl::dot(vr,normal) / (
                            1.0f/mass[iActorA] +
                            1.0f/mass[iActorB] +
                            sycl::dot(
                                sycl::cross(ta, ra) +
                                sycl::cross(tb, rb)
                                , normal
                            )
                        );
                    sycl::float3 addLinVA = -1.0f * normal * impulse / mass[iActorA];
                    sycl::float3 addLinVB = normal * impulse / mass[iActorB];
                    sycl::float3 addAngVA = -1.0f * impulse * ta;
                    sycl::float3 addAngVB = impulse * tb;

                    // os <<    "ra \t= ("    << ra[0] << ',' << ra[1] << ',' << ra[2]
                    //    << ")\nrb \t= ("  << rb[0] << ',' << rb[1] << ',' << rb[2]
                    //    << ")\nnormal \t= ("  << normal[0] << ',' << normal[1] << ',' << normal[2]
                    //    << ")\nvpa \t= (" << vpa[0] << ',' << vpa[1] << ',' << vpa[2]
                    //    << ")\nvpb \t= (" << vpb[0] << ',' << vpb[1] << ',' << vpb[2]
                    //    << ")\nvr \t= ("  << vr[0] << ',' << vr[1] << ',' << vr[2]
                    //    << ")\nta \t= ("  << ta[0] << ',' << ta[1] << ',' << ta[2]
                    //    << ")\ntb \t= ("  << tb[0] << ',' << tb[1] << ',' << tb[2]
                    //    << ")\naddLinVA \t= ("  << addLinVA[0] << ',' << addLinVA[1] << ',' << addLinVA[2]
                    //    << ")\naddLinVB \t= ("  << addLinVB[0] << ',' << addLinVB[1] << ',' << addLinVB[2]
                    //    << ")\naddAngVA \t= ("  << addAngVA[0] << ',' << addAngVA[1] << ',' << addAngVA[2]
                    //    << ")\naddAngVB \t= ("  << addAngVB[0] << ',' << addAngVB[1] << ',' << addAngVB[2]
                    //    << ")\nimpulse \t= " << impulse
                    //    << "\n\n";
                    linearVelocity[iActorA] += addLinVA;
                    linearVelocity[iActorB] += addLinVB;
                    angularVelocity[iActorA] += addAngVA;
                    angularVelocity[iActorB] += addAngVB;
                });
            });
            d2hCopyEvents.insert(d2hCopyEvents.end(),{
                state.linearVelocity.copyToHost(impulseCollisionKernelEvent),
                state.angularVelocity.copyToHost(impulseCollisionKernelEvent),
            });
        } else {
            d2hCopyEvents.insert(d2hCopyEvents.end(),{
                state.linearVelocity.copyToHost(actorKernelEvent),
                state.angularVelocity.copyToHost(actorKernelEvent),
            });
        }

        // Wait for all the device-to-host memory copies to finish
        sycl::event::wait_and_throw(d2hCopyEvents);
    } catch (const std::exception& ex) {
        Corrade::Utility::Error{} << "Exception caught: " << ex.what();
    }

    // Reset force and torque, and transfer serial state data to Actor objects
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
        if ((data.type & Wall::Ymin) > 0 && vy > 0 && vy < 0.1) {
            actors[iActor].addVelocity({0.0f, 0.0001f-1.0f*vy, 0.0f}, {0.0f, 0.0f, 0.0f});
        }
    }
}

} // namespace CollisionSim::Simulation
