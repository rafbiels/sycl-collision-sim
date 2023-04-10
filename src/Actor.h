/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_ACTOR
#define COLLISION_SIM_ACTOR

#include <Magnum/Magnum.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Color.h>

#include <Magnum/MeshTools/Compile.h>
#include <Magnum/MeshTools/Transform.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/Icosphere.h>
#include <Magnum/Primitives/Cylinder.h>
#include <Magnum/Trade/MeshData.h>

namespace CollisionSim {
class Actor {
    public:
        Actor();
        Actor(Magnum::GL::Mesh&& mesh);

        Magnum::GL::Mesh& mesh();

        Magnum::Matrix4& transformation();
        void transformation(Magnum::Matrix4& trf);

        Magnum::Color3& colour();
        void colour(Magnum::Color3& trf);

    private:
        Magnum::GL::Mesh m_mesh;
        Magnum::Matrix4 m_transformation;
        Magnum::Color3 m_colour{0.9, 0.9, 0.9};
};

namespace ActorFactory {
    Actor cube(float scale=1.0);
    Actor sphere(unsigned int subdivisions, float scale=1.0);
    Actor cylinder(unsigned int rings,
                   unsigned int segments,
                   float halfLength,
                   Magnum::Primitives::CylinderFlags flags={},
                   float scale=1.0);
}

} // namespace CollisionSim

#endif // COLLISION_SIM_ACTOR
