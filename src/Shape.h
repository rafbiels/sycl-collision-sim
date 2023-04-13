/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_SHAPE
#define COLLISION_SIM_SHAPE

#include <Magnum/Magnum.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Trade/MeshData.h>

namespace CollisionSim {
class Shape {
    public:
        Shape();
        Shape(Magnum::Trade::MeshData&& mesh);

        Magnum::GL::Mesh& mesh();
        const Magnum::Trade::MeshData& meshData() const;

        Magnum::Matrix4& transformation();
        void transformation(const Magnum::Matrix4& trf);

        Magnum::Color3& colour();
        void colour(const Magnum::Color3& colour);

    private:
        Magnum::Trade::MeshData m_meshData;
        Magnum::GL::Mesh m_mesh;
        Magnum::Matrix4 m_transformation;
        Magnum::Color3 m_colour{0.5f,0.5f,0.5f};
};

} // namespace CollisionSim

#endif // COLLISION_SIM_SHAPE
