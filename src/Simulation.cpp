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

        for (size_t iVertex{0}; iVertex < vertices[0].size(); ++iVertex) {
            if (vertices[0][iVertex] <= min[0]) {
                collision=Wall::Xmin;
                collidingVertexIndex = iVertex;
                normal[0] = 1.0;
                break;
            }
            if (vertices[0][iVertex] >= max[0]) {
                collision=Wall::Xmax;
                collidingVertexIndex = iVertex;
                normal[0] = -1.0;
                break;
            }
            if (vertices[1][iVertex] <= min[1]) {
                collision=Wall::Ymin;
                collidingVertexIndex = iVertex;
                normal[1] = 1.0;
                break;
            }
            if (vertices[1][iVertex] >= max[1]) {
                collision=Wall::Ymax;
                collidingVertexIndex = iVertex;
                normal[1] = -1.0;
                break;
            }
            if (vertices[2][iVertex] <= min[2]) {
                collision=Wall::Zmin;
                collidingVertexIndex = iVertex;
                normal[2] = 1.0;
                break;
            }
            if (vertices[2][iVertex] >= max[2]) {
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

        const Magnum::Vector3 collidingVertexWorld{
            actor.vertexPositionsWorld()[0][collidingVertexIndex],
            actor.vertexPositionsWorld()[1][collidingVertexIndex],
            actor.vertexPositionsWorld()[2][collidingVertexIndex]
        };

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
void collideBroadSequential(std::vector<Actor>& actors, SequentialState* state) {
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
        auto& edges{state->sortedAABBEdges[axis]};
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
        auto& edges{state->sortedAABBEdges[axis]};
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
    state->aabbOverlaps.clear();
    for (const std::pair<uint16_t,uint16_t> overlap : overlaps[0]) {
        if (overlaps[1].contains(overlap) && overlaps[2].contains(overlap)) {
            state->aabbOverlaps.insert(overlap);
        }
    }
}

// -----------------------------------------------------------------------------
void collideNarrowSequential(std::vector<Actor>& actors, SequentialState* state) {
    // ===========================================
    // Algorithm finding the closest vertex-triangle pair from two bodies
    // and comparing against a fixed threshold
    // ===========================================
    for (const auto [iActorA, iActorB] : state->aabbOverlaps) {
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
            sycl::float3 A{verticesB[0][indicesB[3*j]], verticesB[1][indicesB[3*j]], verticesB[2][indicesB[3*j]]};
            sycl::float3 B{verticesB[0][indicesB[3*j+1]], verticesB[1][indicesB[3*j+1]], verticesB[2][indicesB[3*j+1]]};
            sycl::float3 C{verticesB[0][indicesB[3*j+2]], verticesB[1][indicesB[3*j+2]], verticesB[2][indicesB[3*j+2]]};

            // Skip degenerate triangles
            if (Util::equal(A,B) || Util::equal(B,C)) {continue;}

            std::array<sycl::float3,3> triangle{A, B, C};
            const auto closest = Util::closestPointOnTriangle(triangle, verticesA);
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
            sycl::float3 A{verticesA[0][indicesA[3*j]], verticesA[1][indicesA[3*j]], verticesA[2][indicesA[3*j]]};
            sycl::float3 B{verticesA[0][indicesA[3*j+1]], verticesA[1][indicesA[3*j+1]], verticesA[2][indicesA[3*j+1]]};
            sycl::float3 C{verticesA[0][indicesA[3*j+2]], verticesA[1][indicesA[3*j+2]], verticesA[2][indicesA[3*j+2]]};

            // Skip degenerate triangles
            if (Util::equal(A,B) || Util::equal(B,C)) {continue;}

            std::array<sycl::float3,3> triangle{A, B, C};
            const auto closest = Util::closestPointOnTriangle(triangle, verticesB);
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
        if (smallestDistanceSquared < 0.001*0.001) {
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
    a.addVelocity(addLinVA, addAngVA);
    b.addVelocity(addLinVB, addAngVB);
}

// -----------------------------------------------------------------------------
void simulateSequential(float dtime, std::vector<Actor>& actors, SequentialState* state) {
    for (Actor& actor : actors) {
        // Fix floating point loss of orthogonality in the rotation matrix
        Util::orthonormaliseRotation(actor.transformation());
    }
    simulateMotionSequential(dtime, actors);
    collideWorldSequential(actors, state->worldBoundaries);
    collideBroadSequential(actors, state);
    collideNarrowSequential(actors, state);
}

// -----------------------------------------------------------------------------
void simulateParallel(float dtime, std::vector<Actor>& actors, ParallelState* state, sycl::queue* queue) {
    using float3x3 = CollisionSim::ParallelState::float3x3;
    // Copy inputs from Actor objects to serial state data
    for (size_t iActor{0}; iActor<Constants::NumActors; ++iActor) {
        state->linearVelocity.hostContainer[iActor] = Util::toSycl(actors[iActor].linearVelocity());
        state->angularVelocity.hostContainer[iActor] = Util::toSycl(actors[iActor].angularVelocity());
        state->force.hostContainer[iActor] = Util::toSycl(actors[iActor].force());
        state->torque.hostContainer[iActor] = Util::toSycl(actors[iActor].torque());
    }
    try {
        std::vector<sycl::event> h2dCopyEvents{
            state->linearVelocity.copyToDevice(),
            state->angularVelocity.copyToDevice(),
            state->force.copyToDevice(),
            state->torque.copyToDevice()
        };

        // Device pointers to be captured by lambda and copied to device
        // - this is to avoid dereferencing on device the state host pointer
        float* worldBoundaries = state->worldBoundaries.devicePointer;
        float* mass = state->mass.devicePointer;
        uint16_t* actorIndices = state->actorIndices.devicePointer;
        sycl::float3* linearVelocity = state->linearVelocity.devicePointer;
        float3x3* inertiaInv = state->inertiaInv.devicePointer;
        sycl::float3* translation = state->translation.devicePointer;
        sycl::float3* addLinearVelocity = state->addLinearVelocity.devicePointer;
        sycl::float3* addAngularVelocity = state->addAngularVelocity.devicePointer;
        Wall* wallCollisions = state->wallCollisions.devicePointer;
        float3x3* bodyInertiaInv = state->bodyInertiaInv.devicePointer;
        uint16_t* numVertices = state->numVertices.devicePointer;
        uint32_t* verticesOffset = state->verticesOffset.devicePointer;
        std::array<float*,3> bodyVertices = {
            state->bodyVertices[0].devicePointer,
            state->bodyVertices[1].devicePointer,
            state->bodyVertices[2].devicePointer
        };
        std::array<float*,3> worldVertices = {
            state->worldVertices[0].devicePointer,
            state->worldVertices[1].devicePointer,
            state->worldVertices[2].devicePointer
        };
        sycl::float3* angularVelocity = state->angularVelocity.devicePointer;
        float3x3* rotation = state->rotation.devicePointer;
        sycl::float3* force = state->force.devicePointer;
        sycl::float3* torque = state->torque.devicePointer;
        std::array<sycl::float2*,3> aabb {
            state->aabb[0].devicePointer,
            state->aabb[1].devicePointer,
            state->aabb[2].devicePointer
        };
        std::array<Edge*,3> sortedAABBEdges {
            state->sortedAABBEdges[0].devicePointer,
            state->sortedAABBEdges[1].devicePointer,
            state->sortedAABBEdges[2].devicePointer
        };

        sycl::event actorKernelEvent = queue->submit([&](sycl::handler& cgh){
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
        sycl::event vertexKernelEvent = queue->submit([&](sycl::handler& cgh){
            cgh.depends_on(actorKernelEvent);
            cgh.parallel_for<class vertex_kernel>(state->numAllVertices, [=](sycl::id<1> id){
                uint16_t iActor = actorIndices[id];

                // sycl::float3 vertex{
                worldVertices[0][id] =
                    rotation[iActor][0][0]*bodyVertices[0][id] +
                    rotation[iActor][1][0]*bodyVertices[1][id] +
                    rotation[iActor][2][0]*bodyVertices[2][id] +
                    translation[iActor][0];
                worldVertices[1][id] =
                    rotation[iActor][0][1]*bodyVertices[0][id] +
                    rotation[iActor][1][1]*bodyVertices[1][id] +
                    rotation[iActor][2][1]*bodyVertices[2][id] +
                    translation[iActor][1];
                worldVertices[2][id] =
                    rotation[iActor][0][2]*bodyVertices[0][id] +
                    rotation[iActor][1][2]*bodyVertices[1][id] +
                    rotation[iActor][2][2]*bodyVertices[2][id] +
                    translation[iActor][2];
                // };
                sycl::float3 vertex{worldVertices[0][id], worldVertices[1][id], worldVertices[2][id]};

                Wall collision{Wall::None};
                collision |= (WallUnderlyingType{vertex[0] <= worldBoundaries[0]} << 0);
                collision |= (WallUnderlyingType{vertex[0] >= worldBoundaries[1]} << 1);
                collision |= (WallUnderlyingType{vertex[1] <= worldBoundaries[2]} << 2);
                collision |= (WallUnderlyingType{vertex[1] >= worldBoundaries[3]} << 3);
                collision |= (WallUnderlyingType{vertex[2] <= worldBoundaries[4]} << 4);
                collision |= (WallUnderlyingType{vertex[2] >= worldBoundaries[5]} << 5);

                sycl::float3 normal = wallNormal(collision);
                sycl::float3 radius{vertex - translation[iActor]};
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
                wallCollisions[id] = static_cast<Wall>(static_cast<WallUnderlyingType>(collision) * !ignoreAwayFromWall);
            });
        });

        // Calculate the axis-align bounding boxes for each actor
        sycl::event aabbKernelEvent = queue->submit([&](sycl::handler& cgh){
            cgh.depends_on(vertexKernelEvent);
            constexpr static size_t aabbWorkGroupSize{32};
            const sycl::nd_range<1> aabbRange{Constants::NumActors*aabbWorkGroupSize,aabbWorkGroupSize};
            cgh.parallel_for<class aabb_kernel>(aabbRange, [=](sycl::nd_item<1> it){
                #pragma unroll
                for (size_t axis{0}; axis<3; ++axis) {
                    size_t iActor = it.get_group_linear_id();
                    float* begin = worldVertices[axis] + verticesOffset[iActor];
                    float* end = begin + numVertices[iActor];
                    float min = sycl::joint_reduce(it.get_group(), begin, end, sycl::minimum{});
                    float max = sycl::joint_reduce(it.get_group(), begin, end, sycl::maximum{});
                    aabb[axis][iActor][0] = min;
                    aabb[axis][iActor][1] = max;
                }
            });
        });

        // Sort the AABB edges using bitonic sort, following:
        // https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C++SYCL/GraphTraversal/bitonic-sort
        std::optional<sycl::event> aabbSortKernelEvent;
        size_t numSteps = static_cast<size_t>(std::ceil(std::log2(2*Constants::NumActors)));
        size_t gridSize{1ul << numSteps}; // 2**numSteps
        for (size_t step{0}; step<numSteps; ++step) {
            for (int stage=step; stage>=0; --stage) {
                const size_t seq_len{1ul << (stage + 1ul)}; // 2**(stage+1)
                const size_t two_power{1ul << (step - stage)}; // 2**(step-stage)
                aabbSortKernelEvent = queue->submit([&](sycl::handler& cgh){
                    if (aabbSortKernelEvent.has_value()) {
                        cgh.depends_on(aabbSortKernelEvent.value());
                    }
                    cgh.parallel_for<class aabb_sort_kernel>(gridSize, [=](sycl::id<1> id){
                        auto edgeValue = [aabb](unsigned int axis, Edge e) constexpr -> float {
                            return e.isEnd ? aabb[axis][e.actorIndex][1] : aabb[axis][e.actorIndex][0];
                        };
                        auto edgeLess = [aabb, &edgeValue](unsigned int axis, Edge edgeA, Edge edgeB) constexpr -> bool {
                            return edgeValue(axis, edgeA) < edgeValue(axis, edgeB);
                        };
                        auto edgeGreater = [aabb, &edgeValue](unsigned int axis, Edge edgeA, Edge edgeB) constexpr -> bool {
                            return edgeValue(axis, edgeA) > edgeValue(axis, edgeB);
                        };

                        if (id >= 2*Constants::NumActors) {return;}
                        int seq_num = static_cast<int>(id / seq_len);
                        int swapped_elem{-1};
                        int h_len = static_cast<int>(seq_len / 2);
                        if (id < (seq_len*seq_num) + h_len) {
                            swapped_elem = id + h_len;
                        }
                        int odd = static_cast<int>(seq_num / two_power);
                        bool increasing{(odd%2)==0};
                        if (swapped_elem != -1) {
                            if ((edgeGreater(0, sortedAABBEdges[0][id], sortedAABBEdges[0][swapped_elem]) && increasing) ||
                                (edgeLess(0, sortedAABBEdges[0][id], sortedAABBEdges[0][swapped_elem]) && !increasing)) {
                                std::swap(sortedAABBEdges[0][id], sortedAABBEdges[0][swapped_elem]);
                            }
                        }
                    });
                });
            }
        }

        // TODO: clean up what needs to be copied and what it needs to wait for
        std::vector<sycl::event> d2hCopyEvents{
            state->wallCollisions.copyToHost(aabbSortKernelEvent.value()),
            state->addLinearVelocity.copyToHost(aabbSortKernelEvent.value()),
            state->addAngularVelocity.copyToHost(aabbSortKernelEvent.value()),
            state->translation.copyToHost(aabbSortKernelEvent.value()),
            state->rotation.copyToHost(aabbSortKernelEvent.value()),
            state->linearVelocity.copyToHost(aabbSortKernelEvent.value()),
            state->angularVelocity.copyToHost(aabbSortKernelEvent.value()),
            state->aabb[0].copyToHost(aabbSortKernelEvent.value()),
            state->aabb[1].copyToHost(aabbSortKernelEvent.value()),
            state->aabb[2].copyToHost(aabbSortKernelEvent.value()),
            state->sortedAABBEdges[0].copyToHost(aabbSortKernelEvent.value()),
            state->sortedAABBEdges[1].copyToHost(aabbSortKernelEvent.value()),
            state->sortedAABBEdges[2].copyToHost(aabbSortKernelEvent.value()),
        };
        sycl::event::wait_and_throw(d2hCopyEvents);
    } catch (const std::exception& ex) {
        Corrade::Utility::Error{} << "Exception caught: " << ex.what();
    }

    // Corrade::Utility::Debug{} << "Actor 4 AABB:"
    //     << "x=(" << state->aabb[0].hostContainer[4][0] << "," << state->aabb[0].hostContainer[4][1]
    //     << "), y=(" << state->aabb[1].hostContainer[4][0] << "," << state->aabb[1].hostContainer[4][1]
    //     << "), z=(" << state->aabb[2].hostContainer[4][0] << "," << state->aabb[2].hostContainer[4][1]
    //     << ")";

    // Reset force and torque, and transfer serial state data to Actor objects
    for (size_t iActor{0}; iActor<Constants::NumActors; ++iActor) {
        actors[iActor].force({0, 0, 0});
        actors[iActor].torque({0, 0, 0});
        actors[iActor].transformation(Util::transformationMatrix(state->translation.hostContainer[iActor], state->rotation.hostContainer[iActor]));
        actors[iActor].linearVelocity(Util::toMagnum(state->linearVelocity.hostContainer[iActor]));
        actors[iActor].angularVelocity(Util::toMagnum(state->angularVelocity.hostContainer[iActor]));
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
    for (size_t iVertex{0}; iVertex<state->numAllVertices; ++iVertex) {
        Wall collision = state->wallCollisions.hostContainer[iVertex];
        if (collision==Wall::None) {continue;}
        CollisionData& data = actorCollisions[state->actorIndices.hostContainer[iVertex]];
        data.type |= collision;
        data.addLinearV.push_back(state->addLinearVelocity.hostContainer[iVertex]);
        data.addAngularV.push_back(state->addAngularVelocity.hostContainer[iVertex]);
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
