/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "World.h"
#include "Constants.h"
#include <Magnum/Magnum.h>
#include <Magnum/Primitives/Plane.h>

// -----------------------------------------------------------------------------
CollisionSim::World::World(float windowAspectRatio, const Magnum::Vector3& dimensions, float gravity)
: m_dimensions{dimensions}, m_gravity{gravity},
m_walls{Shape{Magnum::Primitives::planeSolid()},
        Shape{Magnum::Primitives::planeSolid()},
        Shape{Magnum::Primitives::planeSolid()},
        Shape{Magnum::Primitives::planeSolid()},
        Shape{Magnum::Primitives::planeSolid()},
        Shape{Magnum::Primitives::planeSolid()}}
{
    using namespace Magnum::Math::Literals;
    m_projection = Magnum::Matrix4::perspectiveProjection(90.0_degf, windowAspectRatio, 0.01, 100.0)
        * Magnum::Matrix4::translation(Magnum::Vector3::zAxis(-1.0*Constants::DefaultWorldDimensions[2]));

    // Aliases for readability
    const float dimX{dimensions[0]};
    const float dimY{dimensions[1]};
    const float dimZ{dimensions[2]};

    // Left wall
    m_walls[0].transformation(
        Magnum::Matrix4::translation({-0.5f*dimX,0.0f,0.0f}) *
        Magnum::Matrix4::rotationY(90.0_degf) *
        Magnum::Matrix4::scaling({0.5f*dimZ,0.5f*dimY,0.0f}));
    // Back wall
    m_walls[1].transformation(
        Magnum::Matrix4::translation({0.0f,0.0f,-0.5f*dimZ}) *
        Magnum::Matrix4::rotationY(0.0_degf) *
        Magnum::Matrix4::scaling({0.5f*dimX,0.5f*dimY,0.0f}));
    // Right wall
    m_walls[2].transformation(
        Magnum::Matrix4::translation({0.5f*dimX,0.0f,0.0f}) *
        Magnum::Matrix4::rotationY(-90.0_degf) *
        Magnum::Matrix4::scaling({0.5f*dimZ,0.5f*dimY,0.0f}));
    // Front wall
    m_walls[3].transformation(
        Magnum::Matrix4::translation({0.0f,0.0f,0.5f*dimZ}) *
        Magnum::Matrix4::rotationY(180.0_degf) *
        Magnum::Matrix4::scaling({0.5f*dimX,0.5f*dimY,0.0f}));
    // Floor
    m_walls[4].transformation(
        Magnum::Matrix4::translation({0.0f,-0.5f*dimY,0.0f}) *
        Magnum::Matrix4::rotationX(-90.0_degf) *
        Magnum::Matrix4::scaling({0.5f*dimX,0.5f*dimZ,0.0f}));
    // Ceiling
    m_walls[5].transformation(
        Magnum::Matrix4::translation({0.0f,0.5f*dimY,0.0f}) *
        Magnum::Matrix4::rotationX(90.0_degf) *
        Magnum::Matrix4::scaling({0.5f*dimX,0.5f*dimZ,0.0f}));

    m_walls[0].colour(0.8f*Magnum::Color3::red());
    m_walls[1].colour(0.8f*Magnum::Color3::green());
    m_walls[2].colour(0.8f*Magnum::Color3::blue());
    m_walls[3].colour(0.8f*Magnum::Color3::yellow());
    m_walls[4].colour(0.8f*Magnum::Color3::magenta());
    m_walls[5].colour(0.8f*Magnum::Color3::cyan());
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
float CollisionSim::World::gravity() const {
    return m_gravity;
}

// -----------------------------------------------------------------------------
std::array<CollisionSim::Shape,CollisionSim::World::NumWalls>& CollisionSim::World::walls() {
    return m_walls;
}
