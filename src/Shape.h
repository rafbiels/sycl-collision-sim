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

        /// Returns SoA vertex positions in body coordinate system
        const std::vector<Magnum::Vector3>& vertexPositions() const;

        /// Returns SoA vertex positions in world coordinate system
        const std::vector<Magnum::Vector3>& vertexPositionsWorld() const;

        /// Returns SoA vertex positions in world coordinate system
        std::vector<Magnum::Vector3>& vertexPositionsWorld_nonconst();

        /// Return the bounding box in world coordinate system
        const Magnum::Range3D& axisAlignedBoundingBox() const;

        /// Recalculate m_vertexPositionsWorld
        void updateVertexPositions();

    private:
        Magnum::Trade::MeshData m_meshData;
        Magnum::GL::Mesh m_mesh;
        Magnum::Matrix4 m_transformation;
        Magnum::Color3 m_colour{0.5f,0.5f,0.5f};
        /// SoA vertex data in body coordinate system
        std::vector<Magnum::Vector3> m_vertexPositions{};
        /// SoA vertex data in world coordinate system
        std::vector<Magnum::Vector3> m_vertexPositionsWorld{};
        /// Axis-aligned bounding box recalculated together with m_vertexPositionsWorld
        Magnum::Range3D m_aabb;
        /// Constant number of vertices, calculated at construction
        size_t m_numVertices{0};
};

} // namespace CollisionSim

#endif // COLLISION_SIM_SHAPE
