/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "CollisionCalculator.h"
#include "Constants.h"
#include "Util.h"
#include "Wall.h"
#include <sycl/sycl.hpp>
#include <limits>
#include <numeric>

class world_collision;

namespace CollisionSim::CollisionCalculator {

void collideWorldSequential(std::vector<Actor>& actors, const Magnum::Range3D& worldBoundaries) {
    int iActor{-1};
    for (auto& actor : actors) {
        ++iActor;
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

void collideWorldParallel(sycl::queue* queue, std::vector<Actor>& actors, State* state) {
    state->load(actors);
    state->resetBuffers(); // FIXME: is there a way to avoid doing this?
    try {
        queue->submit([&](sycl::handler& cgh){
            sycl::accessor boundariesAcc{state->wallCollisionCache().boundariesBuf(), cgh, sycl::read_only};
            sycl::accessor vxAcc{state->vxBuf(), cgh, sycl::read_only};
            sycl::accessor vyAcc{state->vyBuf(), cgh, sycl::read_only};
            sycl::accessor vzAcc{state->vzBuf(), cgh, sycl::read_only};
            sycl::accessor actorIndicesAcc{state->actorIndicesBuf(), cgh, sycl::read_only};
            sycl::accessor massAcc{state->massBuf(), cgh, sycl::read_only};
            sycl::accessor translationAcc{state->translationBuf(), cgh, sycl::read_only};
            sycl::accessor inertiaInvAcc{state->inertiaInvBuf(), cgh, sycl::read_only};
            sycl::accessor linearVelocityAcc{state->linearVelocityBuf(), cgh, sycl::read_only};

            sycl::accessor collisionsAcc{state->wallCollisionCache().collisionsBuf(), cgh, sycl::write_only, sycl::no_init};
            sycl::accessor addLinearVelocityAcc{state->wallCollisionCache().addLinearVelocityBuf(), cgh, sycl::write_only, sycl::no_init};
            sycl::accessor addAngularVelocityAcc{state->wallCollisionCache().addAngularVelocityBuf(), cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for<world_collision>(state->numAllVertices(), [=](sycl::id<1> id){
                Wall collision{Wall::None};
                collision |= (WallUnderlyingType{vxAcc[id] <= boundariesAcc[0]} << 0);
                collision |= (WallUnderlyingType{vxAcc[id] >= boundariesAcc[1]} << 1);
                collision |= (WallUnderlyingType{vyAcc[id] <= boundariesAcc[2]} << 2);
                collision |= (WallUnderlyingType{vyAcc[id] >= boundariesAcc[3]} << 3);
                collision |= (WallUnderlyingType{vzAcc[id] <= boundariesAcc[4]} << 4);
                collision |= (WallUnderlyingType{vzAcc[id] >= boundariesAcc[5]} << 5);
                sycl::float3 normal = wallNormal(collision);
                uint16_t iActor = actorIndicesAcc[id];
                sycl::float3 vertex{vxAcc[id],vyAcc[id],vzAcc[id]};
                sycl::float3 radius{vertex - translationAcc[iActor]};
                sycl::float3 a{sycl::cross(radius, normal)};
                sycl::float3 b{Util::mvmul(inertiaInvAcc[iActor], a)};
                sycl::float3 c{sycl::cross(b, radius)};
                float d{sycl::dot(c, normal)};
                float impulse = (-1.0f - Constants::RestitutionCoefficient) *
                                sycl::dot(linearVelocityAcc[iActor], normal) /
                                (1.0f/massAcc[iActor] + d);
                addLinearVelocityAcc[id] = (impulse / massAcc[iActor]) * normal;
                addAngularVelocityAcc[id] = impulse * b;
                bool ignoreAwayFromWall{sycl::dot(linearVelocityAcc[iActor], normal) > 0.0f};
                collisionsAcc[id] = static_cast<Wall>(static_cast<WallUnderlyingType>(collision) * !ignoreAwayFromWall);
            });
        }).wait_and_throw();
    } catch (const std::exception& ex) {
        Corrade::Utility::Error{} << "Exception caught: " << ex.what();
    }

    struct CollisionData {
        Wall type{Wall::None};
        std::vector<sycl::float3> addLinearV;
        std::vector<sycl::float3> addAngularV;
    };
    std::unordered_map<size_t, CollisionData> actorCollisions; // {actor index, collision data}
    for (size_t iVertex{0}; iVertex<state->numAllVertices(); ++iVertex) {
        Wall collision = state->wallCollisionCache().collisions()[iVertex];
        if (collision==Wall::None) {continue;}
        CollisionData& data = actorCollisions[state->actorIndices()[iVertex]];
        data.type |= collision;
        data.addLinearV.push_back(state->wallCollisionCache().addLinearVelocity()[iVertex]);
        data.addAngularV.push_back(state->wallCollisionCache().addAngularVelocity()[iVertex]);
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

} // namespace CollisionSim::CollisionCalculator
