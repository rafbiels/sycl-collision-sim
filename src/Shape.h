/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_SHAPE
#define COLLISION_SIM_SHAPE

#include <Magnum/Magnum.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Range.h>
#include <Magnum/Trade/MeshData.h>
#include <array>
#include <vector>

namespace CollisionSim {
class Shape {
    public:
        Shape(Magnum::Trade::MeshData&& mesh);

        Magnum::GL::Mesh& mesh();
        const Magnum::Trade::MeshData& meshData() const;

        Magnum::Matrix4& transformation();
        const Magnum::Matrix4& transformation_const() const;
        void transformation(const Magnum::Matrix4& trf);

        Magnum::Color3& colour();
        void colour(const Magnum::Color3& colour);

        size_t numVertices() const;

        size_t numTriangles() const;

        /// Returns vertex positions in body coordinate system
        const std::vector<Magnum::Vector3>& vertexPositions() const;

        /// Returns vertex positions in world coordinate system
        const std::vector<Magnum::Vector3>& vertexPositionsWorld() const;

        /// Returns vertex positions in world coordinate system
        std::vector<Magnum::Vector3>& vertexPositionsWorld_nonconst();

        /// Returns triangle indices array
        const std::vector<Magnum::Vector3ui>& triangles() const;

        /// Return the bounding box in world coordinate system
        const Magnum::Range3D& axisAlignedBoundingBox() const;

        /// Recalculate m_vertexPositionsWorld
        void updateVertexPositions();

    private:
        Magnum::Trade::MeshData m_meshData;
        Magnum::GL::Mesh m_mesh;
        Magnum::Matrix4 m_transformation;
        Magnum::Color3 m_colour{0.5f,0.5f,0.5f};
        /// Vertex data in body coordinate system
        std::vector<Magnum::Vector3> m_vertexPositions{};
        /// Vertex data in world coordinate system
        std::vector<Magnum::Vector3> m_vertexPositionsWorld{};
        /// Triangle indices
        std::vector<Magnum::Vector3ui> m_triangles{};
        /// Axis-aligned bounding box recalculated together with m_vertexPositionsWorld
        Magnum::Range3D m_aabb;
        /// Constant number of vertices, calculated at construction
        size_t m_numVertices{0};
        /// Constant number of triangles, calculated at construction
        size_t m_numTriangles{0};
};

} // namespace CollisionSim

#endif // COLLISION_SIM_SHAPE
