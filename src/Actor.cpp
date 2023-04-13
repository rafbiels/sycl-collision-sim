/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "Actor.h"
#include "Constants.h"
#include "Util.h"
#include <Corrade/Utility/Debug.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Cone.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/Cylinder.h>
#include <Magnum/Primitives/Icosphere.h>
#include <Magnum/Trade/MeshData.h>
#include <limits>

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
    m_mass = Util::round(m_mass);
    m_centreOfMass = Util::round(m_centreOfMass);
    m_covariance = Util::round(m_covariance);

    // Corrade::Utility::Debug{} << "Mass: " << m_mass << ", Centre of mass: " << m_centreOfMass;
    // Not sure about the translation step here
    // Corrade::Utility::Debug{} << "Covariance before translation:\n" << m_covariance;
    auto dx = -1.0f * m_centreOfMass;
    m_covariance += m_mass * (
        Util::outerProduct(dx, Magnum::Vector3{}) +
        Util::outerProduct(Magnum::Vector3{}, dx) +
        Util::outerProduct(dx, dx)
    );
    // Corrade::Utility::Debug{} << "Covariance after translation:\n" << m_covariance;

    m_bodyInertia = Magnum::Matrix3{Magnum::Math::IdentityInit, m_covariance.trace()} - m_covariance;
    m_bodyInertiaInv = m_bodyInertia.inverted();
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
    // Fix floating point loss of orthogonality in the rotation matrix
    // and store the rotation matrix on the stack
    Util::orthonormaliseRotation(transformation());
    Magnum::Matrix3 rotation{transformation().rotation()};

    // ===========================================
    // Rigid body physics simulation based on D. Baraff 2001
    // https://graphics.pixar.com/pbm2001/pdf/notesg.pdf
    // ===========================================
    // Compute linear and angular momentum
    m_linearMomentum += m_force * dtime;
    m_angularMomentum += m_torque * dtime;

    // Corrade::Utility::Debug{} << "p = " << m_linearMomentum << ", L = " << m_angularMomentum;

    // Compute linear and angular velocity
    m_linearVelocity = m_linearMomentum / m_mass;
    m_inertiaInv =  rotation * m_bodyInertiaInv * rotation.transposed();
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
    Magnum::Matrix3 drot = star(m_angularVelocity) * rotation * dtime;
    Magnum::Vector3 dx = m_linearVelocity * dtime;

    // Corrade::Utility::Debug{} << "dx = " << dx << "\ndrot:\n" << drot;

    // Corrade::Utility::Debug{} << "old trf =\n" << m_transformation;

    Magnum::Matrix4 trf{
        {drot[0][0], drot[0][1], drot[0][2], 0.0f},
        {drot[1][0], drot[1][1], drot[1][2], 0.0f},
        {drot[2][0], drot[2][1], drot[2][2], 0.0f},
        {dx[0], dx[1], dx[2], 0.0f},
    };

    transformation(transformation() + trf);
    updateVertexPositions();

    // Corrade::Utility::Debug{} << "new trf =\n" << m_transformation;

    // Corrade::Utility::Debug{} << "new pos = " << m_transformation.translation();
    // Corrade::Utility::Debug{} << "new rot =\n" << m_transformation.rotation();

    // Reset force and torque
    m_force = {0, 0, 0};
    m_torque = {0, 0, 0};
}

// -----------------------------------------------------------------------------
void CollisionSim::Actor::collideWorld(const Magnum::Range3D& boundaries) {
    enum class Collision : short {Min=-1, None=0, Max=1};
    std::array<Collision,3> collisions{Collision::None, Collision::None, Collision::None};
    size_t collidingVertexIndex{std::numeric_limits<size_t>::max()};

    const auto& vertices{vertexPositionsWorld()};
    const Magnum::Vector3& min{boundaries.min()};
    const Magnum::Vector3& max{boundaries.max()};
    Magnum::Vector3 normal{0.0f, 0.0f, 0.0f};

    for (size_t iVertex{0}; iVertex < vertices[0].size(); ++iVertex) {
        if (vertices[0][iVertex] < min[0]) {
            collisions[0]=Collision::Min;
            collidingVertexIndex = iVertex;
            normal[0] = 1.0;
            break;
        }
        if (vertices[0][iVertex] > max[0]) {
            collisions[0]=Collision::Max;
            collidingVertexIndex = iVertex;
            normal[0] = -1.0;
            break;
        }
        if (vertices[1][iVertex] < min[1]) {
            collisions[1]=Collision::Min;
            collidingVertexIndex = iVertex;
            normal[1] = 1.0;
            break;
        }
        if (vertices[1][iVertex] > max[1]) {
            collisions[1]=Collision::Max;
            collidingVertexIndex = iVertex;
            normal[1] = -1.0;
            break;
        }
        if (vertices[2][iVertex] < min[2]) {
            collisions[2]=Collision::Min;
            collidingVertexIndex = iVertex;
            normal[2] = 1.0;
            break;
        }
        if (vertices[2][iVertex] > max[2]) {
            collisions[2]=Collision::Max;
            collidingVertexIndex = iVertex;
            normal[2] = -1.0;
            break;
        }
    }
    if (collisions==std::array<Collision,3>{Collision::None,Collision::None,Collision::None}) {
        return;
    }
    Corrade::Utility::Debug{} << "Collision with world detected, normal = " << normal;
    if (Magnum::Math::dot(m_linearVelocity, normal) > 0.0f) {
        Corrade::Utility::Debug{} << "Velocity points away from the wall, skipping this collision";
        return;
    }
    const Magnum::Vector3 collidingVertexWorld{
        vertexPositionsWorld()[0][collidingVertexIndex],
        vertexPositionsWorld()[1][collidingVertexIndex],
        vertexPositionsWorld()[2][collidingVertexIndex]
    };
    Corrade::Utility::Debug{} << "Before: v = " << m_linearVelocity;
    const Magnum::Vector3 radius = collidingVertexWorld - transformation().translation();
    const auto a = Magnum::Math::cross(radius, normal);
    const auto b = m_inertiaInv * a;
    const auto c = Magnum::Math::cross(b, radius);
    const auto d = Magnum::Math::dot(c, normal);
    float impulse = (-1.0f - Constants::RestitutionCoefficient) * Magnum::Math::dot(m_linearVelocity, normal) / (1.0f/m_mass + d);

    Corrade::Utility::Debug{} << "impulse = " << impulse;

    m_linearVelocity += (impulse / m_mass) * normal;
    m_angularVelocity += impulse * m_inertiaInv * a;

    // TODO: implement better resting condition
    if (normal.y() > 0 && m_linearVelocity.y() > 0 && m_linearVelocity.y() < 0.1) {
        Corrade::Utility::Debug{} << "Resting on the floor, resetting vy to 0";
        m_linearVelocity[1]=0;
    }

    m_linearMomentum = m_mass * m_linearVelocity;
    m_angularMomentum = m_inertiaInv.inverted() * m_angularVelocity;
    Corrade::Utility::Debug{} << "After: v = " << m_linearVelocity;

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
