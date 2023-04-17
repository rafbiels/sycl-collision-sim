/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "CollisionCalculator.h"
#include "Constants.h"
#include <Corrade/Utility/Debug.h>
#include <limits>

namespace CollisionSim::CollisionCalculator {

void collideWorldSequential(std::vector<Actor>& actors, const Magnum::Range3D& worldBoundaries) {
    for (auto& actor : actors) {
        enum class Collision : short {None=-1, Xmin=0, Xmax, Ymin, Ymax, Zmin, Zmax};
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

} // namespace CollisionSim::CollisionCalculator
