/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_ACTOR
#define COLLISION_SIM_ACTOR

#include "Shape.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Matrix3.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Primitives/Cone.h>
#include <Magnum/Primitives/Cylinder.h>
#include <Magnum/Trade/MeshData.h>

namespace CollisionSim {
class Actor : public Shape {
    public:
        Actor(Magnum::Trade::MeshData&& mesh);

        /// Const getters for Actor properties
        ///@{
        float mass() const {return m_mass;}
        const Magnum::Vector3& centreOfMass() const {return m_centreOfMass;}
        const Magnum::Matrix3& covariance() const {return m_covariance;}
        const Magnum::Matrix3& bodyInertiaInv() const {return m_bodyInertiaInv;}
        const Magnum::Matrix3& inertiaInv() const {return m_inertiaInv;}
        const Magnum::Vector3& linearMomentum() const {return m_linearMomentum;}
        const Magnum::Vector3& angularMomentum() const {return m_angularMomentum;}
        const Magnum::Vector3& linearVelocity() const {return m_linearVelocity;}
        const Magnum::Vector3& angularVelocity() const {return m_angularVelocity;}
        const Magnum::Vector3& force() const {return m_force;}
        const Magnum::Vector3& torque() const {return m_torque;}
        ///@}

        /// Direct setters for selected Actor properties
        ///@{
        void inertiaInv(const Magnum::Matrix3& value) {m_inertiaInv=value;}
        void linearMomentum(const Magnum::Vector3& value) {m_linearMomentum=value;}
        void angularMomentum(const Magnum::Vector3& value) {m_angularMomentum=value;}
        void linearVelocity(const Magnum::Vector3& value) {m_linearVelocity=value;}
        void angularVelocity(const Magnum::Vector3& value) {m_angularVelocity=value;}
        void force(const Magnum::Vector3& value) {m_force=value;}
        void torque(const Magnum::Vector3& value) {m_torque=value;}
        ///@}

        /// Apply a force to the centre of mass
        void addForce(const Magnum::Vector3& force);
        /// Apply a force to a given point in the body coordinate system
        void addForce(const Magnum::Vector3& force, const Magnum::Vector3& point);
        /// Add linear and angular velocity and recalculate the momenta
        void addVelocity(const Magnum::Vector3& linear, const Magnum::Vector3& angular);

    private:
        float m_mass{0};
        Magnum::Vector3 m_centreOfMass;
        Magnum::Matrix3 m_covariance;

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
    Actor sphere(float scale, unsigned int subdivisions=2);
    Actor cylinder(float scale,
                   unsigned int rings=4,
                   unsigned int segments=20,
                   float halfLength=1.0,
                   Magnum::Primitives::CylinderFlags flags=Magnum::Primitives::CylinderFlag::CapEnds);
    Actor cone(float scale,
               unsigned int rings=4,
               unsigned int segments=20,
               float halfLength=1.0,
               Magnum::Primitives::ConeFlags flags=Magnum::Primitives::ConeFlag::CapEnd);
}

} // namespace CollisionSim

#endif // COLLISION_SIM_ACTOR
