/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "Actor.h"
#include "Constants.h"
#include <Magnum/MeshTools/Transform.h>
#include <Magnum/Primitives/Cone.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/Cylinder.h>
#include <Magnum/Primitives/Icosphere.h>

namespace CollisionSim::Util {
Magnum::Matrix3 outerProduct(const Magnum::Vector3& a, const Magnum::Vector3& b);
float round(float x);
Magnum::Vector3 round(const Magnum::Vector3& v);
Magnum::Matrix3 round(const Magnum::Matrix3& m);
}

namespace {
Magnum::Trade::MeshData scaledMesh(Magnum::Trade::MeshData&& meshData, float scale) {
    return Magnum::MeshTools::transform3D(
        std::move(meshData),
        Magnum::Matrix4::fromDiagonal({scale, scale, scale, 1.0})
    );
}
}

// -----------------------------------------------------------------------------
CollisionSim::Actor::Actor(Magnum::Trade::MeshData&& meshData)
: Shape{std::move(meshData)} {

    // ===========================================
    // Calculate body inertia matrix by forming tetrahedrons from each face
    // and the origin of the coordinate system. Follow the algorithm from
    // Blow and Binstock 2004, http://number-none.com/blow/inertia/
    // ===========================================
    auto vertices = Shape::meshData().positions3DAsArray();
    auto indices = Shape::meshData().indicesAsArray();
    size_t nFaces{indices.size()/3};

    for (size_t iFace{0}; iFace<nFaces; ++iFace) {
        std::array<Magnum::Vector3,4> tetrahedron{
            Magnum::Vector3{0.0f, 0.0f, 0.0f},
            vertices[indices[3*iFace]],
            vertices[indices[3*iFace+1]],
            vertices[indices[3*iFace+2]],
        };
        Magnum::Matrix3 face{
            vertices[indices[3*iFace]],
            vertices[indices[3*iFace+1]],
            vertices[indices[3*iFace+2]],
        };
        constexpr static float canCovDiag{1.0/60.0f};
        constexpr static float canCovOffdiag{1.0/120.0f};
        constexpr static Magnum::Matrix3 canonicalCovariance = {
            {canCovDiag, canCovOffdiag, canCovOffdiag},
            {canCovOffdiag, canCovDiag, canCovOffdiag},
            {canCovOffdiag, canCovOffdiag, canCovDiag}
        };
        Magnum::Matrix3 covariance = face.determinant() * face * canonicalCovariance * face.transposed();
        Magnum::Vector3 centreOfMass = 0.25f * (tetrahedron[0] + tetrahedron[1] + tetrahedron[2] + tetrahedron[3]);

        constexpr static float density{Constants::DefaultDensity * Constants::Units::Density};
        float volume{(face.determinant()/6.0f) * Constants::Units::Volume};
        float mass{density * volume};

        m_covariance += covariance;
        m_centreOfMass = (m_mass * m_centreOfMass + mass * centreOfMass) / (m_mass + mass);
        m_mass += mass;
    }
    // Round off FP-precision shift
    m_mass = Util::round(m_mass);
    m_centreOfMass = Util::round(m_centreOfMass);
    m_covariance = Util::round(m_covariance);

    // Not sure about the translation step here
    auto dx = -1.0f * m_centreOfMass;
    m_covariance += m_mass * (
        Util::outerProduct(dx, Magnum::Vector3{}) +
        Util::outerProduct(Magnum::Vector3{}, dx) +
        Util::outerProduct(dx, dx)
    );

    Magnum::Matrix3 bodyInertia = Magnum::Matrix3{Magnum::Math::IdentityInit, m_covariance.trace()} - m_covariance;
    m_bodyInertiaInv = bodyInertia.inverted();
    m_inertiaInv = m_bodyInertiaInv;
}

// -----------------------------------------------------------------------------
void CollisionSim::Actor::addForce(const Magnum::Vector3& force) {
    addForce(force, m_centreOfMass);
}

// -----------------------------------------------------------------------------
void CollisionSim::Actor::addForce(const Magnum::Vector3& force, const Magnum::Vector3& point) {
    m_force += force;
    m_torque += Magnum::Math::cross(point, force);
}

// -----------------------------------------------------------------------------
void CollisionSim::Actor::addVelocity(const Magnum::Vector3& linear, const Magnum::Vector3& angular) {
    m_linearVelocity += linear;
    m_angularVelocity += angular;
    m_linearMomentum = m_mass * m_linearVelocity;
    m_angularMomentum = m_inertiaInv.inverted() * m_angularVelocity;
}

// -----------------------------------------------------------------------------
CollisionSim::Actor CollisionSim::ActorFactory::cube(float scale)  {
    return Actor{
        scaledMesh(Magnum::Primitives::cubeSolid(),scale)};
}

// -----------------------------------------------------------------------------
CollisionSim::Actor CollisionSim::ActorFactory::sphere(float scale,
                                                       unsigned int subdivisions) {
    return Actor{
        scaledMesh(Magnum::Primitives::icosphereSolid(subdivisions),scale)};
}

// -----------------------------------------------------------------------------
CollisionSim::Actor CollisionSim::ActorFactory::cylinder(float scale,
                                                         unsigned int rings,
                                                         unsigned int segments,
                                                         float halfLength,
                                                         Magnum::Primitives::CylinderFlags flags) {
    return Actor{
        scaledMesh(Magnum::Primitives::cylinderSolid(rings,segments,halfLength,flags),scale)};
}

// -----------------------------------------------------------------------------
CollisionSim::Actor CollisionSim::ActorFactory::cone(float scale,
                                                     unsigned int rings,
                                                     unsigned int segments,
                                                     float halfLength,
                                                     Magnum::Primitives::ConeFlags flags) {
    return Actor{
        scaledMesh(Magnum::Primitives::coneSolid(rings,segments,halfLength,flags),scale)};
}
