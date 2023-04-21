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
  m_mesh{Magnum::MeshTools::compile(m_meshData)} {
    auto vertices = Shape::meshData().positions3DAsArray();
    m_numVertices = vertices.size();
    m_vertexPositions[0].reserve(vertices.size());
    m_vertexPositions[1].reserve(vertices.size());
    m_vertexPositions[2].reserve(vertices.size());
    for (const auto& vertex : vertices) {
        m_vertexPositions[0].push_back(vertex[0]);
        m_vertexPositions[1].push_back(vertex[1]);
        m_vertexPositions[2].push_back(vertex[2]);
    }
    m_vertexPositionsWorld = m_vertexPositions;
}

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
const Magnum::Matrix4& CollisionSim::Shape::transformation_const() const {
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

// -----------------------------------------------------------------------------
size_t CollisionSim::Shape::numVertices() const {
    return m_numVertices;
}

// -----------------------------------------------------------------------------
const std::array<std::vector<float>,3>& CollisionSim::Shape::vertexPositions() const {
    return m_vertexPositions;
}

// -----------------------------------------------------------------------------
const std::array<std::vector<float>,3>& CollisionSim::Shape::vertexPositionsWorld() const {
    return m_vertexPositionsWorld;
}

// -----------------------------------------------------------------------------
Magnum::Range3D CollisionSim::Shape::axisAlignedBoundingBox() {
    Magnum::Range3D box{
        {m_vertexPositionsWorld[0][0], m_vertexPositionsWorld[1][0],m_vertexPositionsWorld[2][0]},
        {m_vertexPositionsWorld[0][0], m_vertexPositionsWorld[1][0],m_vertexPositionsWorld[2][0]}
    };
    for (size_t iVertex{1}; iVertex<m_vertexPositionsWorld[0].size(); ++iVertex) {
        for (size_t axis{0}; axis<3; ++axis) {
            if (m_vertexPositionsWorld[axis][iVertex] < box.min()[axis]) {
                box.min()[axis] = m_vertexPositionsWorld[axis][iVertex];
            }
            if (m_vertexPositionsWorld[axis][iVertex] > box.max()[axis]) {
                box.max()[axis] = m_vertexPositionsWorld[axis][iVertex];
            }
        }
    }
    return box;
}

// -----------------------------------------------------------------------------
void CollisionSim::Shape::updateVertexPositions() {
    for (size_t iVertex{0}; iVertex<m_vertexPositions[0].size(); ++iVertex) {
        m_vertexPositionsWorld[0][iVertex] =
            m_transformation[0][0]*m_vertexPositions[0][iVertex] +
            m_transformation[1][0]*m_vertexPositions[1][iVertex] +
            m_transformation[2][0]*m_vertexPositions[2][iVertex] +
            m_transformation[3][0];
        m_vertexPositionsWorld[1][iVertex] =
            m_transformation[0][1]*m_vertexPositions[0][iVertex] +
            m_transformation[1][1]*m_vertexPositions[1][iVertex] +
            m_transformation[2][1]*m_vertexPositions[2][iVertex] +
            m_transformation[3][1];
        m_vertexPositionsWorld[2][iVertex] =
            m_transformation[0][2]*m_vertexPositions[0][iVertex] +
            m_transformation[1][2]*m_vertexPositions[1][iVertex] +
            m_transformation[2][2]*m_vertexPositions[2][iVertex] +
            m_transformation[3][2];
    }
}
