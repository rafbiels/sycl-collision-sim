/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "Shape.h"
#include <Magnum/MeshTools/Compile.h>
#include <limits>

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
std::array<std::vector<float>,3>& CollisionSim::Shape::vertexPositionsWorld_nonconst() {
    return m_vertexPositionsWorld;
}

// -----------------------------------------------------------------------------
const Magnum::Range3D& CollisionSim::Shape::axisAlignedBoundingBox() const {
    return m_aabb;
}

// -----------------------------------------------------------------------------
void CollisionSim::Shape::updateVertexPositions() {
    m_aabb = {
        Magnum::Vector3{std::numeric_limits<float>::max()},
        Magnum::Vector3{std::numeric_limits<float>::min()}
    };
    for (size_t iVertex{0}; iVertex<m_vertexPositions[0].size(); ++iVertex) {
        for (size_t axis{0}; axis<3; ++axis) {
            m_vertexPositionsWorld[axis][iVertex] =
                m_transformation[0][axis]*m_vertexPositions[0][iVertex] +
                m_transformation[1][axis]*m_vertexPositions[1][iVertex] +
                m_transformation[2][axis]*m_vertexPositions[2][iVertex] +
                m_transformation[3][axis];
            m_aabb.min()[axis] = std::min(m_aabb.min()[axis], m_vertexPositionsWorld[axis][iVertex]);
            m_aabb.max()[axis] = std::max(m_aabb.max()[axis], m_vertexPositionsWorld[axis][iVertex]);
        }
    }
}
