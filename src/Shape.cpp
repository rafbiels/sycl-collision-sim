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
    const auto& vertices{m_meshData.positions3DAsArray()};
    m_numVertices = vertices.size();
    m_vertexPositions = std::vector<Magnum::Vector3>(vertices.begin(), vertices.end());
    m_vertexPositionsWorld = std::vector<Magnum::Vector3>(vertices.begin(), vertices.end());

    // Fill the triangle indices only for indexed meshes
    if (!m_meshData.isIndexed()) {return;}
    const auto& triangles{m_meshData.indicesAsArray()};
    m_numTriangles = triangles.size() / 3;
    m_triangles.reserve(m_numTriangles);
    for (size_t iTriangle{0}; iTriangle < m_numTriangles; ++iTriangle) {
        m_triangles.emplace_back(triangles[3*iTriangle], triangles[3*iTriangle+1], triangles[3*iTriangle+2]);
    }
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
size_t CollisionSim::Shape::numTriangles() const {
    return m_numTriangles;
}

// -----------------------------------------------------------------------------
const std::vector<Magnum::Vector3>& CollisionSim::Shape::vertexPositions() const {
    return m_vertexPositions;
}

// -----------------------------------------------------------------------------
const std::vector<Magnum::Vector3>& CollisionSim::Shape::vertexPositionsWorld() const {
    return m_vertexPositionsWorld;
}

// -----------------------------------------------------------------------------
std::vector<Magnum::Vector3>& CollisionSim::Shape::vertexPositionsWorld_nonconst() {
    return m_vertexPositionsWorld;
}

// -----------------------------------------------------------------------------
const std::vector<Magnum::Vector3ui>& CollisionSim::Shape::triangles() const {
    return m_triangles;
}

// -----------------------------------------------------------------------------
const Magnum::Range3D& CollisionSim::Shape::axisAlignedBoundingBox() const {
    return m_aabb;
}

// -----------------------------------------------------------------------------
void CollisionSim::Shape::updateVertexPositions() {
    m_aabb = {
        Magnum::Vector3{std::numeric_limits<float>::max()},
        Magnum::Vector3{std::numeric_limits<float>::lowest()}
    };
    for (size_t iVertex{0}; iVertex<m_vertexPositions.size(); ++iVertex) {
        for (size_t axis{0}; axis<3; ++axis) {
            m_vertexPositionsWorld[iVertex][axis] =
                m_transformation[0][axis]*m_vertexPositions[iVertex][0] +
                m_transformation[1][axis]*m_vertexPositions[iVertex][1] +
                m_transformation[2][axis]*m_vertexPositions[iVertex][2] +
                m_transformation[3][axis];
            m_aabb.min()[axis] = std::min(m_aabb.min()[axis], m_vertexPositionsWorld[iVertex][axis]);
            m_aabb.max()[axis] = std::max(m_aabb.max()[axis], m_vertexPositionsWorld[iVertex][axis]);
        }
    }
}
