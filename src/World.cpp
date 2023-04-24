/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "World.h"
#include "Constants.h"
#include <Magnum/Primitives/Plane.h>

// -----------------------------------------------------------------------------
CollisionSim::World::World(float windowAspectRatio, const Magnum::Vector3& dimensions, float gravity)
: m_boundaries{Magnum::Range3D::fromCenter({}, dimensions)}, m_gravity{gravity},
m_walls{Shape{Magnum::Primitives::planeSolid()},
        Shape{Magnum::Primitives::planeSolid()},
        Shape{Magnum::Primitives::planeSolid()},
        Shape{Magnum::Primitives::planeSolid()},
        Shape{Magnum::Primitives::planeSolid()},
        Shape{Magnum::Primitives::planeSolid()}}
{
    using namespace Magnum::Math::Literals;
    m_projection = Magnum::Matrix4::perspectiveProjection(90.0_degf, windowAspectRatio, 0.01, 100.0)
        * Magnum::Matrix4::translation(Magnum::Vector3::zAxis(-1.7*Constants::DefaultWorldDimensions[2]))
        * Magnum::Matrix4::rotationY(-20.0_degf);

    // Alias for readability
    const Magnum::Range3D& b{m_boundaries};

    // Left wall
    m_walls[0].transformation(
        Magnum::Matrix4::translation({b.left(),0.0f,0.0f}) *
        Magnum::Matrix4::rotationY(90.0_degf) *
        Magnum::Matrix4::scaling({0.5f*b.sizeZ(),0.5f*b.sizeY(),1.0f}));
    // Back wall
    m_walls[1].transformation(
        Magnum::Matrix4::translation({0.0f,0.0f,b.back()}) *
        Magnum::Matrix4::rotationY(0.0_degf) *
        Magnum::Matrix4::scaling({0.5f*b.sizeX(),0.5f*b.sizeY(),1.0f}));
    // Right wall
    m_walls[2].transformation(
        Magnum::Matrix4::translation({b.right(),0.0f,0.0f}) *
        Magnum::Matrix4::rotationY(-90.0_degf) *
        Magnum::Matrix4::scaling({0.5f*b.sizeZ(),0.5f*b.sizeY(),1.0f}));
    // Front wall
    m_walls[3].transformation(
        Magnum::Matrix4::translation({0.0f,0.0f,b.front()}) *
        Magnum::Matrix4::rotationY(180.0_degf) *
        Magnum::Matrix4::scaling({0.5f*b.sizeX(),0.5f*b.sizeY(),1.0f}));
    // Floor
    m_walls[4].transformation(
        Magnum::Matrix4::translation({0.0f,b.bottom(),0.0f}) *
        Magnum::Matrix4::rotationX(-90.0_degf) *
        Magnum::Matrix4::scaling({0.5f*b.sizeX(),0.5f*b.sizeZ(),1.0f}));
    // Ceiling
    m_walls[5].transformation(
        Magnum::Matrix4::translation({0.0f,b.top(),0.0f}) *
        Magnum::Matrix4::rotationX(90.0_degf) *
        Magnum::Matrix4::scaling({0.5f*b.sizeX(),0.5f*b.sizeZ(),1.0f}));

    m_walls[0].colour(Magnum::Vector3{0.45f});
    m_walls[1].colour(Magnum::Vector3{0.45f});
    m_walls[2].colour(Magnum::Vector3{0.45f});
    m_walls[3].colour(Magnum::Vector3{0.45f});
    m_walls[4].colour(Magnum::Vector3{0.45f});
    m_walls[5].colour(Magnum::Vector3{0.45f});
}

// -----------------------------------------------------------------------------
Magnum::Matrix4& CollisionSim::World::projection() {
    return m_projection;
}

// -----------------------------------------------------------------------------
void CollisionSim::World::projection(Magnum::Matrix4& proj) {
    m_projection = proj;
}

// -----------------------------------------------------------------------------
const Magnum::Range3D& CollisionSim::World::boundaries() const {
    return m_boundaries;
}

// -----------------------------------------------------------------------------
float CollisionSim::World::gravity() const {
    return m_gravity;
}

// -----------------------------------------------------------------------------
std::array<CollisionSim::Shape,CollisionSim::World::NumWalls>& CollisionSim::World::walls() {
    return m_walls;
}
