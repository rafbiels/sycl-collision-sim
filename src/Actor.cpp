/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "Actor.h"
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/Cylinder.h>
#include <Magnum/Primitives/Icosphere.h>
#include <Magnum/Trade/MeshData.h>

namespace {
Magnum::Trade::MeshData scaledMesh(Magnum::Trade::MeshData&& meshData, float scale) {
    return Magnum::MeshTools::transform3D(
        std::move(meshData),
        Magnum::Matrix4::fromDiagonal({scale, scale, scale, 1.0})
    );
}
}

// -----------------------------------------------------------------------------
CollisionSim::Actor::Actor(Magnum::GL::Mesh&& mesh)
: m_mesh(std::move(mesh)) {}

// -----------------------------------------------------------------------------
Magnum::GL::Mesh& CollisionSim::Actor::mesh() {
    return m_mesh;
}

// -----------------------------------------------------------------------------
Magnum::Matrix4& CollisionSim::Actor::transformation() {
    return m_transformation;
}

// -----------------------------------------------------------------------------
void CollisionSim::Actor::transformation(Magnum::Matrix4& trf) {
    m_transformation = trf;
}

// -----------------------------------------------------------------------------
Magnum::Color3& CollisionSim::Actor::colour() {
    return m_colour;
}

// -----------------------------------------------------------------------------
void CollisionSim::Actor::colour(Magnum::Color3& trf) {
    m_colour = trf;
}

// -----------------------------------------------------------------------------
CollisionSim::Actor CollisionSim::ActorFactory::cube(float scale)  {
    return Actor{Magnum::MeshTools::compile(
        scaledMesh(Magnum::Primitives::cubeSolid(),scale))};
}

// -----------------------------------------------------------------------------
CollisionSim::Actor CollisionSim::ActorFactory::sphere(unsigned int subdivisions,
                                                       float scale) {
    return Actor{Magnum::MeshTools::compile(
        scaledMesh(Magnum::Primitives::icosphereSolid(subdivisions),scale))};
}

// -----------------------------------------------------------------------------
CollisionSim::Actor CollisionSim::ActorFactory::cylinder(unsigned int rings,
                                                         unsigned int segments,
                                                         float halfLength,
                                                         Magnum::Primitives::CylinderFlags flags,
                                                         float scale) {
    return Actor{Magnum::MeshTools::compile(
        scaledMesh(Magnum::Primitives::cylinderSolid(rings,segments,halfLength,flags),scale))};
}
