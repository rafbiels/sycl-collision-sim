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

class world_collision;

namespace {
enum class WallCollision : uint8_t {
    None = 0,
    Xmin = 1,
    Xmax = 1<<1,
    Ymin = 1<<2,
    Ymax = 1<<3,
    Zmin = 1<<4,
    Zmax = 1<<5
};
constexpr WallCollision& operator|=(WallCollision& a, uint8_t b) {
    a = static_cast<WallCollision>(static_cast<uint8_t>(a) | b);
    return a;
}
constexpr WallCollision& operator|=(WallCollision& a, WallCollision b) {
    a = static_cast<WallCollision>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
    return a;
}
constexpr uint8_t operator&(WallCollision a, WallCollision b) {
    return static_cast<uint8_t>(a) & static_cast<uint8_t>(b);
}
constexpr sycl::float3 wallNormal(WallCollision collision) {
    return {((collision & WallCollision::Xmax) >> 1) - ((collision & WallCollision::Xmin) >> 0),
            ((collision & WallCollision::Ymax) >> 3) - ((collision & WallCollision::Ymin) >> 2),
            ((collision & WallCollision::Zmax) >> 5) - ((collision & WallCollision::Zmin) >> 4)};
}
constexpr sycl::float3 toSycl(const Magnum::Vector3& vec) {
    const float (&data)[3] = vec.data();
    return sycl::float3{data[0],data[1],data[2]};
}
constexpr std::array<sycl::float3,3> toSycl(const Magnum::Matrix3& mat) {
    const float (&data)[9] = mat.data();
    return {sycl::float3{data[0],data[1],data[2]},
            sycl::float3{data[3],data[4],data[5]},
            sycl::float3{data[6],data[7],data[8]}};
}
constexpr Magnum::Vector3 toMagnum(const sycl::float3& vec) {
    return Magnum::Vector3{vec[0],vec[1],vec[2]};
}
constexpr sycl::float3 mvmul(const std::array<sycl::float3,3>& mat, const sycl::float3& vec) {
    return {mat[0][0]*vec[0] + mat[1][0]*vec[1] + mat[2][0]*vec[2],
            mat[0][1]*vec[0] + mat[1][1]*vec[1] + mat[2][1]*vec[2],
            mat[0][2]*vec[0] + mat[1][2]*vec[1] + mat[2][2]*vec[2]};
}
}

namespace CollisionSim::CollisionCalculator {

void collideWorldSequential(std::vector<Actor>& actors, const Magnum::Range3D& worldBoundaries) {
    int iActor{-1};
    for (auto& actor : actors) {
        ++iActor;
        WallCollision collision{WallCollision::None};
        size_t collidingVertexIndex{std::numeric_limits<size_t>::max()};

        const auto& vertices{actor.vertexPositionsWorld()};
        const Magnum::Vector3& min{worldBoundaries.min()};
        const Magnum::Vector3& max{worldBoundaries.max()};
        Magnum::Vector3 normal{0.0f, 0.0f, 0.0f};

        for (size_t iVertex{0}; iVertex < vertices[0].size(); ++iVertex) {
            if (vertices[0][iVertex] <= min[0]) {
                collision=WallCollision::Xmin;
                collidingVertexIndex = iVertex;
                normal[0] = 1.0;
                break;
            }
            if (vertices[0][iVertex] >= max[0]) {
                collision=WallCollision::Xmax;
                collidingVertexIndex = iVertex;
                normal[0] = -1.0;
                break;
            }
            if (vertices[1][iVertex] <= min[1]) {
                collision=WallCollision::Ymin;
                collidingVertexIndex = iVertex;
                normal[1] = 1.0;
                break;
            }
            if (vertices[1][iVertex] >= max[1]) {
                collision=WallCollision::Ymax;
                collidingVertexIndex = iVertex;
                normal[1] = -1.0;
                break;
            }
            if (vertices[2][iVertex] <= min[2]) {
                collision=WallCollision::Zmin;
                collidingVertexIndex = iVertex;
                normal[2] = 1.0;
                break;
            }
            if (vertices[2][iVertex] >= max[2]) {
                collision=WallCollision::Zmax;
                collidingVertexIndex = iVertex;
                normal[2] = -1.0;
                break;
            }
        }
        if (collision==WallCollision::None) {continue;}
        // Corrade::Utility::Debug{} << "WallCollision with world detected, normal = " << normal;
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
        const float impulse = (-1.0f - Constants::RestitutionCoefficient) *
                              Magnum::Math::dot(actor.linearVelocity(), normal) /
                              (1.0f/actor.mass() + d);

        // Corrade::Utility::Debug{} << "impulse = " << impulse;
        Magnum::Vector3 addLinearV = (impulse / actor.mass()) * normal;
        Magnum::Vector3 addAngularV = impulse * b;
        actor.addVelocity(addLinearV, addAngularV);
        /*
        Corrade::Utility::Debug{} << "[CPU] actor " << iActor
                                  << " collision type " << static_cast<uint8_t>(collision)
                                  << ", results = ("
                                  << addLinearV[0] << ","
                                  << addLinearV[1] << ","
                                  << addLinearV[2] << "), ("
                                  << addAngularV[0] << ","
                                  << addAngularV[1] << ","
                                  << addAngularV[2] << ")";
        */
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

    // Linearise data of all actors
    using float3x3 = std::array<sycl::float3,3>;
    std::array<std::vector<float>,3> allVertices;
    std::vector<uint16_t> actorIndices; // Caution: restricting numActors to 65536
    std::vector<float> mass;
    std::vector<sycl::float3> translation(numActors, sycl::float3{0.0f});
    std::vector<float3x3> inertiaInv(numActors, float3x3{});
    std::vector<sycl::float3> linearVelocity(numActors, sycl::float3{0.0f});
    std::vector<sycl::float3> addLinearVelocity(numAllVertices, sycl::float3{0.0f});
    std::vector<sycl::float3> addAngularVelocity(numAllVertices, sycl::float3{0.0f});

    actorIndices.reserve(numAllVertices);
    allVertices[0].reserve(numAllVertices);
    allVertices[1].reserve(numAllVertices);
    allVertices[2].reserve(numAllVertices);
    mass.reserve(numActors);
    for (size_t iActor{0}; iActor<numActors; ++iActor) {
        const auto& vertices = actors[iActor].vertexPositionsWorld();
        actorIndices.insert(actorIndices.end(), vertices[0].size(), static_cast<uint16_t>(iActor));
        allVertices[0].insert(allVertices[0].end(), vertices[0].begin(), vertices[0].end());
        allVertices[1].insert(allVertices[1].end(), vertices[1].begin(), vertices[1].end());
        allVertices[2].insert(allVertices[2].end(), vertices[2].begin(), vertices[2].end());
        mass.push_back(actors[iActor].mass());
        translation[iActor] = toSycl(actors[iActor].transformation().translation());
        inertiaInv[iActor] = toSycl(actors[iActor].inertiaInv());
        linearVelocity[iActor] = toSycl(actors[iActor].linearVelocity());
    }

    std::vector<WallCollision> collisions(numAllVertices, WallCollision::None);

    sycl::queue queue{sycl::gpu_selector_v};
    try {
        sycl::buffer<float,1> boundariesBuf{boundaries.data(), 6};
        sycl::buffer<float,1> vxBuf{allVertices[0]};
        sycl::buffer<float,1> vyBuf{allVertices[1]};
        sycl::buffer<float,1> vzBuf{allVertices[2]};
        sycl::buffer<WallCollision,1> collisionsBuf{collisions};
        sycl::buffer<uint16_t,1> actorIndicesBuf{actorIndices};
        sycl::buffer<float,1> massBuf{mass};
        sycl::buffer<sycl::float3,1> translationBuf{translation};
        sycl::buffer<float3x3,1> intertiaInvBuf{inertiaInv};
        sycl::buffer<sycl::float3,1> linearVelocityBuf{linearVelocity};
        sycl::buffer<sycl::float3,1> addLinearVelocityBuf{addLinearVelocity};
        sycl::buffer<sycl::float3,1> addAngularVelocityBuf{addAngularVelocity};
        queue.submit([&](sycl::handler& cgh){
            sycl::accessor boundariesAcc{boundariesBuf, cgh, sycl::read_only};
            sycl::accessor vxAcc{vxBuf, cgh, sycl::read_only};
            sycl::accessor vyAcc{vyBuf, cgh, sycl::read_only};
            sycl::accessor vzAcc{vzBuf, cgh, sycl::read_only};
            sycl::accessor actorIndicesAcc{actorIndicesBuf, cgh, sycl::read_only};
            sycl::accessor massAcc{massBuf, cgh, sycl::read_only};
            sycl::accessor translationAcc{translationBuf, cgh, sycl::read_only};
            sycl::accessor inertiaInvAcc{intertiaInvBuf, cgh, sycl::read_only};
            sycl::accessor linearVelocityAcc{linearVelocityBuf, cgh, sycl::read_only};
            sycl::accessor addLinearVelocityAcc{addLinearVelocityBuf, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor addAngularVelocityAcc{addAngularVelocityBuf, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor collisionsAcc{collisionsBuf, cgh, sycl::write_only, sycl::no_init};
            auto os = sycl::stream{65536, 1024, cgh};
            cgh.parallel_for<world_collision>(numAllVertices, [=](sycl::id<1> id){
                // WallCollision collision{WallCollision::None};
                collisionsAcc[id] |= (uint8_t{vxAcc[id] <= boundariesAcc[0]} << 0);
                collisionsAcc[id] |= (uint8_t{vxAcc[id] >= boundariesAcc[1]} << 1);
                collisionsAcc[id] |= (uint8_t{vyAcc[id] <= boundariesAcc[2]} << 2);
                collisionsAcc[id] |= (uint8_t{vyAcc[id] >= boundariesAcc[3]} << 3);
                collisionsAcc[id] |= (uint8_t{vzAcc[id] <= boundariesAcc[4]} << 4);
                collisionsAcc[id] |= (uint8_t{vzAcc[id] >= boundariesAcc[5]} << 5);
                // collisionsAcc[id] = collision;
                sycl::float3 normal = wallNormal(collisionsAcc[id]);
                uint16_t iActor = actorIndicesAcc[id];
                sycl::float3 vertex{vxAcc[id],vyAcc[id],vzAcc[id]};
                sycl::float3 radius{vertex - translationAcc[iActor]};
                sycl::float3 a{sycl::cross(radius, normal)};
                sycl::float3 b{mvmul(inertiaInvAcc[iActor], a)};
                sycl::float3 c{sycl::cross(b, radius)};
                float d{sycl::dot(c, normal)};
                float impulse = (-1.0f - Constants::RestitutionCoefficient) *
                                sycl::dot(linearVelocityAcc[iActor], normal) /
                                (1.0f/massAcc[iActor] + d);
                addLinearVelocityAcc[id] = (impulse / massAcc[iActor]) * normal;
                addAngularVelocityAcc[id] = impulse * b;
                bool ignoreAwayFromWall{sycl::dot(addLinearVelocityAcc[id], normal) > 0.0f};
                // branchless version of: if (ignoreAwayFromWall) {collisionsAcc[id]=Collision::None;}
                collisionsAcc[id] = static_cast<WallCollision>(static_cast<uint8_t>(collisionsAcc[id]) * !ignoreAwayFromWall);
            });
        });
        queue.wait_and_throw();
    } catch (const std::exception& ex) {
        Corrade::Utility::Error{} << "Exception caught: " << ex.what();
    }
    struct CollisionData {
        WallCollision type{WallCollision::None};
        std::vector<sycl::float3> addLinearV;
        std::vector<sycl::float3> addAngularV;
    };
    std::unordered_map<size_t, CollisionData> actorCollisions; // {actor index, collision data}
    for (size_t iVertex{0}; iVertex<numAllVertices; ++iVertex) {
        if (collisions[iVertex]==WallCollision::None) {continue;}
        CollisionData& data = actorCollisions[actorIndices[iVertex]];
        data.type |= collisions[iVertex];
        data.addLinearV.push_back(addLinearVelocity[iVertex]);
        data.addAngularV.push_back(addAngularVelocity[iVertex]);
    }
    for (const auto& [iActor, data] : actorCollisions) {
        size_t num{data.addLinearV.size()};
        auto accumulateMean = [&num](const sycl::float3& a, const sycl::float3& b){
            return a + b/static_cast<float>(num);
        };
        sycl::float3 meanAddLinV = std::accumulate(data.addLinearV.begin(),data.addLinearV.end(),sycl::float3{0.0f},accumulateMean);
        sycl::float3 meanAddAngV = std::accumulate(data.addAngularV.begin(),data.addAngularV.end(),sycl::float3{0.0f},accumulateMean);
        /*
        Corrade::Utility::Debug{} << "[GPU] actor " << iActor
                                  << " collision type " << static_cast<uint8_t>(data.type)
                                  << ", results = ("
                                  << meanAddLinV[0] << ","
                                  << meanAddLinV[1] << ","
                                  << meanAddLinV[2] << "), ("
                                  << meanAddAngV[0] << ","
                                  << meanAddAngV[1] << ","
                                  << meanAddAngV[2] << ")";
        */
        actors[iActor].addVelocity(toMagnum(meanAddLinV), toMagnum(meanAddAngV));
        const float vy{actors[iActor].linearVelocity().y()};
        // TODO: implement better resting condition
        if ((data.type & WallCollision::Ymin) > 0 && vy > 0 && vy < 0.1) {
            // Corrade::Utility::Debug{} << "Resting on the floor, resetting vy to 0";
            actors[iActor].addVelocity({0.0f, -1.0f*vy, 0.0f}, {0.0f, 0.0f, 0.0f});
        }
    }
}

} // namespace CollisionSim::CollisionCalculator
