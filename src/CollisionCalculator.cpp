/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "CollisionCalculator.h"
#include "Constants.h"
#include <Corrade/Utility/Debug.h>
#include <sycl/sycl.hpp>
#include <limits>
#include <numeric>

class world_collision_detect;

namespace {
enum class Collision : uint8_t {
    None = 0,
    Xmin = 1,
    Xmax = 1<<1,
    Ymin = 1<<2,
    Ymax = 1<<3,
    Zmin = 1<<4,
    Zmax = 1<<5
};
}

namespace CollisionSim::CollisionCalculator {

void collideWorldSequential(std::vector<Actor>& actors, const Magnum::Range3D& worldBoundaries) {
    int iActor{-1};
    for (auto& actor : actors) {
        ++iActor;
        Collision collision{Collision::None};
        size_t collidingVertexIndex{std::numeric_limits<size_t>::max()};

        const auto& vertices{actor.vertexPositionsWorld()};
        const Magnum::Vector3& min{worldBoundaries.min()};
        const Magnum::Vector3& max{worldBoundaries.max()};
        Magnum::Vector3 normal{0.0f, 0.0f, 0.0f};

        for (size_t iVertex{0}; iVertex < vertices[0].size(); ++iVertex) {
            if (vertices[0][iVertex] <= min[0]) {
                collision=Collision::Xmin;
                collidingVertexIndex = iVertex;
                normal[0] = 1.0;
                break;
            }
            if (vertices[0][iVertex] >= max[0]) {
                collision=Collision::Xmax;
                collidingVertexIndex = iVertex;
                normal[0] = -1.0;
                break;
            }
            if (vertices[1][iVertex] <= min[1]) {
                collision=Collision::Ymin;
                collidingVertexIndex = iVertex;
                normal[1] = 1.0;
                break;
            }
            if (vertices[1][iVertex] >= max[1]) {
                collision=Collision::Ymax;
                collidingVertexIndex = iVertex;
                normal[1] = -1.0;
                break;
            }
            if (vertices[2][iVertex] <= min[2]) {
                collision=Collision::Zmin;
                collidingVertexIndex = iVertex;
                normal[2] = 1.0;
                break;
            }
            if (vertices[2][iVertex] >= max[2]) {
                collision=Collision::Zmax;
                collidingVertexIndex = iVertex;
                normal[2] = -1.0;
                break;
            }
        }
        if (collision==Collision::None) {continue;}
        Corrade::Utility::Debug{} << "[CPU] actor " << iActor << " collision type " << static_cast<uint8_t>(collision);
        // Corrade::Utility::Debug{} << "Collision with world detected, normal = " << normal;
        if (Magnum::Math::dot(actor.linearVelocity(), normal) > 0.0f) {
            // Corrade::Utility::Debug{} << "Velocity " << actor.linearVelocity() << " points away from the wall, skipping this collision";
            continue;
        }
        const Magnum::Vector3 collidingVertexWorld{
            actor.vertexPositionsWorld()[0][collidingVertexIndex],
            actor.vertexPositionsWorld()[1][collidingVertexIndex],
            actor.vertexPositionsWorld()[2][collidingVertexIndex]
        };
        // Corrade::Utility::Debug{} << "Before: v = " << actor.linearVelocity();
        const Magnum::Vector3 radius = collidingVertexWorld - actor.transformation().translation();
        const auto a = Magnum::Math::cross(radius, normal);
        const auto b = actor.inertiaInv() * a;
        const auto c = Magnum::Math::cross(b, radius);
        const auto d = Magnum::Math::dot(c, normal);
        float impulse = (-1.0f - Constants::RestitutionCoefficient) * Magnum::Math::dot(actor.linearVelocity(), normal) / (1.0f/actor.mass() + d);

        // Corrade::Utility::Debug{} << "impulse = " << impulse;
        actor.addVelocity((impulse / actor.mass()) * normal, impulse * actor.inertiaInv() * a);

        const float vy{actor.linearVelocity().y()};
        // TODO: implement better resting condition
        if (normal.y() > 0 && vy > 0 && vy < 0.1) {
            // Corrade::Utility::Debug{} << "Resting on the floor, resetting vy to 0";
            actor.addVelocity({0.0f, -1.0f*vy, 0.0f}, {0.0f, 0.0f, 0.0f});
        }
        // Corrade::Utility::Debug{} << "After: v = " << actor.linearVelocity();
    }
}

void collideWorldParallel(std::vector<Actor>& actors, const Magnum::Range3D& worldBoundaries, size_t numAllVertices) {
    const std::array<float,6> boundaries{
        worldBoundaries.min()[0], worldBoundaries.max()[0],
        worldBoundaries.min()[1], worldBoundaries.max()[1],
        worldBoundaries.min()[2], worldBoundaries.max()[2],
    };
    const size_t numActors{actors.size()};

    std::array<std::vector<float>,3> allVertices;
    std::vector<size_t> actorIndices;
    actorIndices.reserve(numAllVertices);
    allVertices[0].reserve(numAllVertices);
    allVertices[1].reserve(numAllVertices);
    allVertices[2].reserve(numAllVertices);
    for (size_t iActor{0}; iActor<numActors; ++iActor) {
        const auto& vertices = actors[iActor].vertexPositionsWorld();
        actorIndices.insert(actorIndices.end(), vertices[0].size(), iActor);
        allVertices[0].insert(allVertices[0].end(), vertices[0].begin(), vertices[0].end());
        allVertices[1].insert(allVertices[1].end(), vertices[1].begin(), vertices[1].end());
        allVertices[2].insert(allVertices[2].end(), vertices[2].begin(), vertices[2].end());
    }

    std::vector<uint8_t> collisions(numAllVertices, static_cast<uint8_t>(Collision::None));

    sycl::queue queue{sycl::gpu_selector_v};
    try {
        sycl::buffer<float,1> boundariesBuf{boundaries.data(), 6};
        sycl::buffer<float,1> vxBuf{allVertices[0].data(), numAllVertices};
        sycl::buffer<float,1> vyBuf{allVertices[1].data(), numAllVertices};
        sycl::buffer<float,1> vzBuf{allVertices[2].data(), numAllVertices};
        sycl::buffer<uint8_t,1> collisionsBuf{collisions.data(), numAllVertices};
        queue.submit([&](sycl::handler& cgh){
            sycl::accessor boundariesAcc{boundariesBuf, cgh, sycl::read_only};
            sycl::accessor vxAcc{vxBuf, cgh, sycl::read_only};
            sycl::accessor vyAcc{vyBuf, cgh, sycl::read_only};
            sycl::accessor vzAcc{vzBuf, cgh, sycl::read_only};
            sycl::accessor collisionsAcc{collisionsBuf, cgh, sycl::write_only, sycl::no_init};
            cgh.parallel_for<world_collision_detect>(numAllVertices, [=](sycl::id<1> id){
                collisionsAcc[id] |= (uint8_t{vxAcc[id] <= boundariesAcc[0]} << 0);
                collisionsAcc[id] |= (uint8_t{vxAcc[id] >= boundariesAcc[1]} << 1);
                collisionsAcc[id] |= (uint8_t{vyAcc[id] <= boundariesAcc[2]} << 2);
                collisionsAcc[id] |= (uint8_t{vyAcc[id] >= boundariesAcc[3]} << 3);
                collisionsAcc[id] |= (uint8_t{vzAcc[id] <= boundariesAcc[4]} << 4);
                collisionsAcc[id] |= (uint8_t{vzAcc[id] >= boundariesAcc[5]} << 5);
            });
        });
        queue.wait_and_throw();
    } catch (const std::exception& ex) {
        Corrade::Utility::Error{} << "Exception caught: " << ex.what();
    }
    std::unordered_map<size_t, uint8_t> actorCollisions; // {actor index, collision type}
    for (size_t iVertex{0}; iVertex<numAllVertices; ++iVertex) {
        if (collisions[iVertex]==0) {continue;}
        actorCollisions[actorIndices[iVertex]] |= collisions[iVertex];
    }
    for (const auto& [iActor,collision] : actorCollisions) {
        Corrade::Utility::Debug{} << "[GPU] actor " << iActor << " collision type " << static_cast<uint8_t>(collision);
    }
}

} // namespace CollisionSim::CollisionCalculator
