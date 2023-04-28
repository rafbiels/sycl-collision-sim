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

class actor_kernel;
class vertex_kernel;

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
void simulateSequential(float dtime, std::vector<Actor>& actors, const Magnum::Range3D& worldBoundaries) {
    for (Actor& actor : actors) {
        // Fix floating point loss of orthogonality in the rotation matrix
        Util::orthonormaliseRotation(actor.transformation());
    }
    simulateMotionSequential(dtime, actors);
    collideWorldSequential(actors, worldBoundaries);
}

// -----------------------------------------------------------------------------
void simulateParallel(float dtime, sycl::queue* queue, std::vector<Actor>& actors, State* state) {
    using float3x3 = CollisionSim::State::float3x3;
    // Copy inputs from Actor objects to serial state data
    for (size_t iActor{0}; iActor<state->numActors; ++iActor) {
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
        std::array<float*,3> bodyVertices = {
            state->bodyVertices[0].devicePointer,
            state->bodyVertices[1].devicePointer,
            state->bodyVertices[2].devicePointer
        };
        sycl::float3* angularVelocity = state->angularVelocity.devicePointer;
        float3x3* rotation = state->rotation.devicePointer;
        sycl::float3* force = state->force.devicePointer;
        sycl::float3* torque = state->torque.devicePointer;

        sycl::event actorKernelEvent = queue->submit([&](sycl::handler& cgh){
            cgh.depends_on(h2dCopyEvents);
            cgh.parallel_for<actor_kernel>(state->numActors, [=](sycl::id<1> id){
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
            cgh.parallel_for<vertex_kernel>(state->numAllVertices, [=](sycl::id<1> id){
                uint16_t iActor = actorIndices[id];

                sycl::float3 vertex{
                    rotation[iActor][0][0]*bodyVertices[0][id] +
                    rotation[iActor][1][0]*bodyVertices[1][id] +
                    rotation[iActor][2][0]*bodyVertices[2][id] +
                    translation[iActor][0],
                    rotation[iActor][0][1]*bodyVertices[0][id] +
                    rotation[iActor][1][1]*bodyVertices[1][id] +
                    rotation[iActor][2][1]*bodyVertices[2][id] +
                    translation[iActor][1],
                    rotation[iActor][0][2]*bodyVertices[0][id] +
                    rotation[iActor][1][2]*bodyVertices[1][id] +
                    rotation[iActor][2][2]*bodyVertices[2][id] +
                    translation[iActor][2]
                };

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

        std::vector<sycl::event> d2hCopyEvents{
            state->wallCollisions.copyToHost(vertexKernelEvent),
            state->addLinearVelocity.copyToHost(vertexKernelEvent),
            state->addAngularVelocity.copyToHost(vertexKernelEvent),
            state->translation.copyToHost(vertexKernelEvent),
            state->rotation.copyToHost(vertexKernelEvent),
            state->linearVelocity.copyToHost(vertexKernelEvent),
            state->angularVelocity.copyToHost(vertexKernelEvent)
        };
        sycl::event::wait_and_throw(d2hCopyEvents);
    } catch (const std::exception& ex) {
        Corrade::Utility::Error{} << "Exception caught: " << ex.what();
    }

    // Reset force and torque, and transfer serial state data to Actor objects
    for (size_t iActor{0}; iActor<state->numActors; ++iActor) {
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
