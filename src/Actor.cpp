/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "Actor.h"
#include "Constants.h"
#include "Util.h"
#include <Corrade/Utility/Debug.h>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Tags.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Cone.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/Cylinder.h>
#include <Magnum/Primitives/Icosphere.h>
#include <Magnum/Trade/MeshData.h>

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
: m_meshData{std::move(meshData)} {
    m_mesh = Magnum::MeshTools::compile(m_meshData);

    // ===========================================
    // Calculate body inertia matrix by forming tetrahedrons from each face
    // and the origin of the coordinate system. Follow the algorithm from
    // Blow and Binstock 2004, http://number-none.com/blow/inertia/
    // ===========================================
    auto vertices = m_meshData.positions3DAsArray();
    auto indices = m_meshData.indicesAsArray();
    size_t nFaces{indices.size()/3};
    // Corrade::Utility::Debug{} << "Printing Actor meshdata " << vertices.size() << " vertices:";
    // for (const auto& v : vertices) {
    //     Corrade::Utility::Debug{} << v;
    // }
    // Corrade::Utility::Debug{} << "Printing Actor meshdata " << indices.size() << " indices:";
    // for (const auto& idx : indices) {
    //     Corrade::Utility::Debug{} << idx;
    // }
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
    constexpr static float precision{1e6f};
    auto myRound = [](float x){
        if (x < 1.0/precision && x > -1.0/precision) {return 0.0f;}
        return std::round(precision*x)/precision;
    };
    m_mass = myRound(m_mass);
    m_centreOfMass = {
        myRound(m_centreOfMass[0]),
        myRound(m_centreOfMass[1]),
        myRound(m_centreOfMass[2]),
    };
    m_covariance = {
        {myRound(m_covariance[0][0]), myRound(m_covariance[0][1]), myRound(m_covariance[0][2])},
        {myRound(m_covariance[1][0]), myRound(m_covariance[1][1]), myRound(m_covariance[1][2])},
        {myRound(m_covariance[2][0]), myRound(m_covariance[2][1]), myRound(m_covariance[2][2])},
    };
    Corrade::Utility::Debug{} << "Mass: " << m_mass << ", Centre of mass: " << m_centreOfMass;
    // Not sure about the translation step here
    Corrade::Utility::Debug{} << "Covariance before translation:\n" << m_covariance;
    auto dx = -1.0f * m_centreOfMass;
    m_covariance += m_mass * (
        Util::outerProduct(dx, Magnum::Vector3{}) +
        Util::outerProduct(Magnum::Vector3{}, dx) +
        Util::outerProduct(dx, dx)
    );
    Corrade::Utility::Debug{} << "Covariance after translation:\n" << m_covariance;

    m_bodyInertia = Magnum::Matrix3{Magnum::Math::IdentityInit, m_covariance.trace()} - m_covariance;
    m_bodyInertiaInv = m_bodyInertia.inverted();
}

// -----------------------------------------------------------------------------
Magnum::GL::Mesh& CollisionSim::Actor::mesh() {
    return m_mesh;
}

// -----------------------------------------------------------------------------
Magnum::Matrix4& CollisionSim::Actor::transformation() {
    return m_transformation;
}

// -----------------------------------------------------------------------------
void CollisionSim::Actor::transformation(const Magnum::Matrix4& trf) {
    m_transformation = trf;
}

// -----------------------------------------------------------------------------
Magnum::Color3& CollisionSim::Actor::colour() {
    return m_colour;
}

// -----------------------------------------------------------------------------
void CollisionSim::Actor::colour(Magnum::Color3& trf) {
    m_colour = trf;
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
void CollisionSim::Actor::computeState(float dtime) {
    // Compute linear and angular momentum
    m_linearMomentum += m_force * dtime;
    m_angularMomentum += m_torque * dtime;

    // Corrade::Utility::Debug{} << "p = " << m_linearMomentum << ", L = " << m_angularMomentum;

    // Compute linear and angular velocity
    m_linearVelocity = m_linearMomentum / m_mass;
    m_inertiaInv =  m_transformation.rotation() * m_bodyInertiaInv * m_transformation.rotation().transposed();
    m_angularVelocity = m_inertiaInv * m_angularMomentum;

    // Corrade::Utility::Debug{} << "v = " << m_linearVelocity << ", omega = " << m_angularVelocity;

    // Apply translation and rotation
    auto star = [](const Magnum::Vector3& v) {
        return Magnum::Matrix3{
            { 0.0f,  v[2], -v[1]},
            {-v[2],  0.0f,  v[0]},
            { v[1], -v[0],  0.0f}
        };
    };
    Magnum::Matrix3 drot = star(m_angularVelocity) * m_transformation.rotation() * dtime;
    Magnum::Vector3 dx = m_linearVelocity * dtime;

    // Corrade::Utility::Debug{} << "dx = " << dx << "\ndrot:\n" << drot;

    // Corrade::Utility::Debug{} << "old trf =\n" << m_transformation;

    Magnum::Matrix4 trf{
        {drot[0][0], drot[0][1], drot[0][2], 0.0f},
        {drot[1][0], drot[1][1], drot[1][2], 0.0f},
        {drot[2][0], drot[2][1], drot[2][2], 0.0f},
        {dx[0], dx[1], dx[2], 0.0f},
    };

    m_transformation = m_transformation + trf;

    // Corrade::Utility::Debug{} << "new trf =\n" << m_transformation;

    // Corrade::Utility::Debug{} << "new pos = " << m_transformation.translation();
    // Corrade::Utility::Debug{} << "new rot =\n" << m_transformation.rotation();

    // Reset force and torque
    m_force = {0, 0, 0};
    m_torque = {0, 0, 0};
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
