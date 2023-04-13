/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_ACTOR
#define COLLISION_SIM_ACTOR

#include "Shape.h"
#include <Magnum/Magnum.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Range.h>
#include <Magnum/Math/Color.h>

#include <Magnum/MeshTools/Compile.h>
#include <Magnum/MeshTools/Transform.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/Icosphere.h>
#include <Magnum/Primitives/Cylinder.h>
#include <Magnum/Primitives/Cone.h>
#include <Magnum/Trade/MeshData.h>

namespace CollisionSim {
class Actor : public Shape {
    public:
        Actor(Magnum::Trade::MeshData&& mesh);

        float mass() const {return m_mass;}

        /// Apply a force to the centre of mass
        void addForce(const Magnum::Vector3& force);
        /// Apply a force to a given point in the body coordinate system
        void addForce(const Magnum::Vector3& force, const Magnum::Vector3& point);
        /// Process forces and state to compute new state after time \c dtime
        void computeState(float dtime);
        /// Check world boundary collisions and apply the collision physics
        void collideWorld(const Magnum::Range3D& boundaries);

    private:
        float m_mass{0};
        Magnum::Vector3 m_centreOfMass;
        Magnum::Matrix3 m_covariance;

        Magnum::Matrix3 m_bodyInertia;
        Magnum::Matrix3 m_bodyInertiaInv;
        Magnum::Matrix3 m_inertiaInv;
        Magnum::Vector3 m_linearMomentum;
        Magnum::Vector3 m_angularMomentum;
        Magnum::Vector3 m_linearVelocity;
        Magnum::Vector3 m_angularVelocity;
        Magnum::Vector3 m_force;
        Magnum::Vector3 m_torque;
};

namespace ActorFactory {
    Actor cube(float scale);
    Actor sphere(float scale, unsigned int subdivisions);
    Actor cylinder(float scale,
                   unsigned int rings,
                   unsigned int segments,
                   float halfLength,
                   Magnum::Primitives::CylinderFlags flags=Magnum::Primitives::CylinderFlag::CapEnds);
    Actor cone(float scale,
               unsigned int rings,
               unsigned int segments,
               float halfLength,
               Magnum::Primitives::ConeFlags flags=Magnum::Primitives::ConeFlag::CapEnd);
}

} // namespace CollisionSim

#endif // COLLISION_SIM_ACTOR
