/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "Shape.h"
#include <Magnum/MeshTools/Compile.h>

// -----------------------------------------------------------------------------
CollisionSim::Shape::Shape(Magnum::Trade::MeshData&& meshData)
: m_meshData{std::move(meshData)},
  m_mesh{Magnum::MeshTools::compile(m_meshData)} {}

// -----------------------------------------------------------------------------
Magnum::GL::Mesh& CollisionSim::Shape::mesh() {
    return m_mesh;
}

// -----------------------------------------------------------------------------
const Magnum::Trade::MeshData& CollisionSim::Shape::meshData() const {
    return m_meshData;
}

// -----------------------------------------------------------------------------
Magnum::Matrix4& CollisionSim::Shape::transformation() {
    return m_transformation;
}

// -----------------------------------------------------------------------------
void CollisionSim::Shape::transformation(const Magnum::Matrix4& trf) {
    m_transformation = trf;
}

// -----------------------------------------------------------------------------
Magnum::Color3& CollisionSim::Shape::colour() {
    return m_colour;
}

// -----------------------------------------------------------------------------
void CollisionSim::Shape::colour(const Magnum::Color3& colour) {
    m_colour = colour;
}
