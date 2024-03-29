/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "Util.h"
#include <Magnum/Math/Algorithms/GramSchmidt.h>
#include <stdexcept>

namespace CollisionSim::Util {

// -----------------------------------------------------------------------------
void Timer::reset() {
    m_currentTime = clock_t::now();
}

// -----------------------------------------------------------------------------
duration_t Timer::step() {
    time_point_t previousTime = m_currentTime;
    m_currentTime = clock_t::now();
    return m_currentTime - previousTime;
}

// -----------------------------------------------------------------------------
duration_t Timer::peek() const {
    return clock_t::now() - m_currentTime;
}

// -----------------------------------------------------------------------------
bool Timer::stepIfElapsed(duration_t duration) {
    if (peek() < duration) {return false;}
    reset();
    return true;
}

// -----------------------------------------------------------------------------
RepeatTask::RepeatTask(std::function<void()>&& callback)
: m_callback(std::move(callback)) {}

// -----------------------------------------------------------------------------
RepeatTask::~RepeatTask() {
    stop();
}

// -----------------------------------------------------------------------------
void RepeatTask::start(duration_t interval) {
    if (m_thread!=nullptr) {
        throw std::runtime_error("RepeatTask::start called on already running task");
    }
    m_interval = interval;
    m_keepRunning = true;
    m_timer.reset();
    m_thread = std::make_unique<std::thread>([this]{run();});
}

// -----------------------------------------------------------------------------
void RepeatTask::stop() {
    m_keepRunning = false;
    if (m_thread==nullptr) {return;}
    m_thread->join();
    m_thread.reset();
}

// -----------------------------------------------------------------------------
void RepeatTask::run() {
    while (m_keepRunning) {
        duration_t timeLeft{m_interval - m_timer.peek()};
        if (timeLeft < duration_t{0}) {
            m_timer.reset();
            m_callback();
        } else {
            std::this_thread::sleep_for(timeLeft);
        }
    }
}

// -----------------------------------------------------------------------------
Magnum::Matrix3 outerProduct(const Magnum::Vector3& a, const Magnum::Vector3& b) {
    return Magnum::Matrix3{ // construct from *column* vectors
        {a[0]*b[0], a[1]*b[0], a[2]*b[0]},
        {a[0]*b[1], a[1]*b[1], a[2]*b[1]},
        {a[0]*b[2], a[1]*b[2], a[2]*b[2]}
    };
}

// -----------------------------------------------------------------------------
float round(float x) {
    if (x < 1.0/RoundingPrecision && x > -1.0/RoundingPrecision) {return 0.0f;}
    return std::round(RoundingPrecision*x)/RoundingPrecision;
}

// -----------------------------------------------------------------------------
Magnum::Vector3 round(const Magnum::Vector3& v) {
    return Magnum::Vector3{
        round(v[0]), round(v[1]), round(v[2])
    };
}

// -----------------------------------------------------------------------------
Magnum::Vector4 round(const Magnum::Vector4& v) {
    return Magnum::Vector4{
        round(v[0]), round(v[1]), round(v[2]), round(v[3])
    };
}

// -----------------------------------------------------------------------------
Magnum::Matrix3 round(const Magnum::Matrix3& m) {
    return Magnum::Matrix3{
        round(m[0]), round(m[1]), round(m[2])
    };
}

// -----------------------------------------------------------------------------
Magnum::Matrix4 round(const Magnum::Matrix4& m) {
    return Magnum::Matrix4{
        round(m[0]), round(m[1]), round(m[2]), round(m[3])
    };
}

// -----------------------------------------------------------------------------
void orthonormaliseRotation(Magnum::Matrix4 &trfMatrix) {
    Magnum::Matrix3 rot{
        Magnum::Vector3{trfMatrix[0][0], trfMatrix[0][1], trfMatrix[0][2]}.normalized(),
        Magnum::Vector3{trfMatrix[1][0], trfMatrix[1][1], trfMatrix[1][2]}.normalized(),
        Magnum::Vector3{trfMatrix[2][0], trfMatrix[2][1], trfMatrix[2][2]}.normalized()
    };
    if (rot.isOrthogonal()) {return;}
    Magnum::Math::Algorithms::gramSchmidtOrthonormalizeInPlace(rot);
    trfMatrix[0] = {rot[0][0], rot[0][1], rot[0][2], 0.0f};
    trfMatrix[1] = {rot[1][0], rot[1][1], rot[1][2], 0.0f};
    trfMatrix[2] = {rot[2][0], rot[2][1], rot[2][2], 0.0f};
}

// -----------------------------------------------------------------------------
sycl::nd_range<1> ndRangeAllCU(size_t minNumWorkItems, const sycl::queue& queue) {
    size_t nWG{queue.get_device().get_info<sycl::info::device::max_compute_units>()};
    size_t maxWGSize{queue.get_device().get_info<sycl::info::device::max_work_group_size>()};
    size_t wgSize{(minNumWorkItems+nWG-1)/nWG};
    while (nWG*wgSize < minNumWorkItems || wgSize > maxWGSize) {
        nWG *= 2;
        wgSize = (minNumWorkItems+nWG-1)/nWG;
    }
    return sycl::nd_range<1>{nWG*wgSize, wgSize};
}

// -----------------------------------------------------------------------------
SYCL_EXTERNAL
std::array<std::array<sycl::float3,3>,3> triangleTransform(const std::array<sycl::float3,3>& triangle) {
    // ===========================================
    // Part of the "2D Method" following M.W. Jones 1995
    // 3D Distance from a Point to a Triangle
    // http://www-compsci.swan.ac.uk/~csmark/PDFS/1995_3D_distance_point_to_triangle
    // ===========================================

    using float3x3 = std::array<sycl::float3,3>;
    sycl::float3 A{0.0f};
    sycl::float3 B{triangle[1]-triangle[0]};
    sycl::float3 C{triangle[2]-triangle[0]};

    // Rotate the problem around x-axis such that B lies in the xz plane
    const float denomRotX = sycl::sqrt(B[1]*B[1]+B[2]*B[2]);
    const float sinRotX = denomRotX==0 ? 0.0f : sycl::copysign(B[1] / denomRotX, B[2]);
    const float cosRotX = denomRotX==0 ? 1.0f : sycl::copysign(B[2] / denomRotX, B[1]);
    const float3x3 rotX = {
        sycl::float3{1.0f, 0.0f, 0.0f},
        sycl::float3{0.0f, cosRotX, sinRotX},
        sycl::float3{0.0f, -sinRotX, cosRotX}
    };
    B = mvmul(rotX, B);
    C = mvmul(rotX, C);

    // Rotate the problem around y-axis such that B lies on the z-axis
    const float denomRotY = sycl::sqrt(B[0]*B[0]+B[2]*B[2]);
    const float sinRotY = denomRotY==0 ? 0.0f : sycl::copysign(B[0] / denomRotY, B[2]);
    const float cosRotY = denomRotY==0 ? 1.0f : sycl::copysign(B[2] / denomRotY, -B[0]);
    const float3x3 rotY = {
        sycl::float3{cosRotY, 0.0f, -sinRotY},
        sycl::float3{0.0f, 1.0f, 0.0f},
        sycl::float3{sinRotY, 0.0f, cosRotY}
    };
    B = mvmul(rotY, B);
    C = mvmul(rotY, C);

    // Rotate the problem around z-axis such that C lies in the yz plane
    const float denomRotZ = sycl::sqrt(C[0]*C[0]+C[1]*C[1]);
    const float sinRotZ = denomRotZ==0 ? 0.0f : sycl::copysign(C[0] / denomRotZ, C[0]);
    const float cosRotZ = denomRotZ==0 ? 1.0f : sycl::copysign(C[1] / denomRotZ, -C[1]);
    const float3x3 rotZ = {
        sycl::float3{cosRotZ, -sinRotZ, 0.0f},
        sycl::float3{sinRotZ, cosRotZ, 0.0f},
        sycl::float3{0.0f, 0.0f, 1.0f}
    };
    B = mvmul(rotZ, B);
    C = mvmul(rotZ, C);

    // Rotation matrix to transform vertices into the new coordinate system
    float3x3 rot{
        sycl::float3{
            cosRotZ*cosRotY,
            -sinRotZ*cosRotY,
            -sinRotY
        },
        sycl::float3{
            sinRotZ*cosRotX+cosRotZ*sinRotY*sinRotX,
            cosRotZ*cosRotX-sinRotZ*sinRotY*sinRotX,
            cosRotY*sinRotX
        },
        sycl::float3{
            -sinRotZ*sinRotX+cosRotZ*sinRotY*cosRotX,
            -cosRotZ*sinRotX-sinRotZ*sinRotY*cosRotX,
            cosRotY*cosRotX
        }
    };

    // Rotation matrix to transform vertices back to the original coordinate system
    // Each angle is the opposite of the previous transform: cos(-a)=cos(a), sin(-a)=-sin(a)
    float3x3 negRot{
        sycl::float3{cosRotY*cosRotZ,
                     sinRotX*sinRotY*cosRotZ+cosRotX*sinRotZ,
                     cosRotX*sinRotY*cosRotZ-sinRotX*sinRotZ},
        sycl::float3{-cosRotY*sinRotZ,
                     -sinRotX*sinRotY*sinRotZ+cosRotX*cosRotZ,
                     -cosRotX*sinRotY*sinRotZ-sinRotX*cosRotZ},
        sycl::float3{-sinRotY,
                     sinRotX*cosRotY,
                     cosRotX*cosRotY}
    };

    return {std::array<sycl::float3,3>{A, B, C}, rot, negRot};
}

// -----------------------------------------------------------------------------
SYCL_EXTERNAL
std::pair<sycl::float3,float> closestPointOnTriangle(const std::array<sycl::float3,3>& triangle, const sycl::float3& point) {
    // ===========================================
    // Part of the "2D Method" following M.W. Jones 1995
    // 3D Distance from a Point to a Triangle
    // http://www-compsci.swan.ac.uk/~csmark/PDFS/1995_3D_distance_point_to_triangle
    // ===========================================
    // Note the computations assume a coordinate system where A is (0,0,0), B is (0,0,bz), C is (0,cy,cz)
    const sycl::float3& A{triangle[0]};
    const sycl::float3& B{triangle[1]};
    const sycl::float3& C{triangle[2]};
    const sycl::float3& P{point};

    const float forceAnticlockwise = (-B[2]*C[1] < 0) ? -1.0f : 1.0f;
    const float edgeAB = forceAnticlockwise * P[1] * B[2];
    const float edgeBC = forceAnticlockwise * (P[1]*(C[2]-B[2]) - C[1]*(P[2]-B[2]));
    const float edgeCA = forceAnticlockwise * (C[1]*P[2] - C[2]*P[1]);

    float distanceSquared{std::numeric_limits<float>::lowest()};
    sycl::float3 closestPoint{0.0f, 0.0f, 0.0f};

    if (edgeAB < 0 && edgeBC < 0 && edgeCA < 0) {
        // - - -
        // 2D point inside triangle
        distanceSquared = P[0]*P[0];
        closestPoint = sycl::float3{0.0f, P[1], P[2]};
    } else if (edgeAB >= 0) {
        if (edgeCA >= 0) {
            // + - + (edgeBC assumed -)
            // 2D point closest to triangle vertex A
            // if (edgeBC > 0) {throw std::runtime_error("edgeBA expected <= 0 but is > 0");}
            distanceSquared = P[0]*P[0]+P[1]*P[1]+P[2]*P[2];
            closestPoint = A;
        } else if (edgeBC >= 0) {
            // + + -
            // 2D point closest to triangle vertex B
            distanceSquared = P[0]*P[0] + P[1]*P[1] + (P[2]-B[2])*(P[2]-B[2]);
            closestPoint = B;
        } else {
            // + - -
            // 2D point closest to triangle edge AB
            // AB lies on z-axis, so d(P, AB) is  d(P, z-axis)
            const float distanceYZSquared = P[1]*P[1];
            distanceSquared = P[0]*P[0] + distanceYZSquared;
            const float ADOverAB = sycl::sqrt(
                (P[1]*P[1] + P[2]*P[2] - distanceYZSquared) / (B[2]*B[2])
            );
            closestPoint = sycl::float3{
                0.0f,
                0.0f,
                (B[2])*ADOverAB,
            };
        }
    } else if (edgeBC >= 0) {
        if (edgeCA >= 0) {
            // - + +
            // 2D point closest to triangle vertex C
            distanceSquared = P[0]*P[0] + (P[1]-C[1])*(P[1]-C[1]) + (P[2]-C[2])*(P[2]-C[2]);
            closestPoint = C;
        } else {
            // - + -
            // 2D point closest to triangle edge BC
            const float BmPz = B[2]-P[2];
            const float CmBz = C[2]-B[2];
            const float distanceYZSquared = (C[1]*BmPz + P[1]*CmBz) * (C[1]*BmPz + P[1]*CmBz) / (C[1]*C[1] + CmBz*CmBz);
            distanceSquared = P[0]*P[0] + distanceYZSquared;
            const float BDOverBC = sycl::sqrt(
                (P[1]*P[1] + (P[2]-B[2])*(P[2]-B[2]) - distanceYZSquared) /
                (C[1]*C[1] + (C[2]-B[2])*(C[2]-B[2]))
            );
            closestPoint = sycl::float3{
                0.0f,
                C[1]*BDOverBC,
                B[2] + (C[2]-B[2])*BDOverBC,
            };
        }
    } else {
        // - - + (edgeCA assumed +)
        // 2D point closest to triangle edge CA
        // if (edgeCA < 0) {throw std::runtime_error("edgeCA expected >= 0 but is < 0");}
        const float distanceYZSquared = (C[2]*P[1]-C[1]*P[2]) * (C[2]*P[1]-C[1]*P[2]) / (C[1]*C[1]+C[2]*C[2]);
        distanceSquared = P[0]*P[0] + distanceYZSquared;
        const float CDOverCA = sycl::sqrt(
            ((P[1]-C[1])*(P[1]-C[1]) + (P[2]-C[2])*(P[2]-C[2]) - distanceYZSquared) /
            (C[1]*C[1] + C[2]*C[2])
        );
        closestPoint = sycl::float3{
            0.0f,
            C[1] - C[1]*CDOverCA,
            C[2] - C[2]*CDOverCA,
        };
    }

    return {closestPoint, distanceSquared};
}

// -----------------------------------------------------------------------------
ClosestPointOnTriangleReturnValue closestPointOnTriangleND(const std::array<sycl::float3,3>& triangle, const std::vector<Magnum::Vector3>& vertices) {
    using float3x3 = std::array<sycl::float3,3>;
    std::array<float3x3,3> transformed = triangleTransform(triangle);
    const float3x3& rot{transformed[1]};
    const float3x3& negRot{transformed[2]};

    float smallestDistanceSquared{std::numeric_limits<float>::max()};
    sycl::float3 bestPointOnTriangle{0.0f, 0.0f, 0.0f};
    size_t bestVertexIndex{std::numeric_limits<size_t>::max()};

    for (size_t iVertex{0}; iVertex<vertices.size(); ++iVertex) {
        sycl::float3 P{Util::toSycl(vertices[iVertex])};
        P -= triangle[0];
        P = mvmul(rot,P);

        const auto [closestPoint, distanceSquared] = closestPointOnTriangle(transformed[0], P);

        if (distanceSquared < smallestDistanceSquared) {
            smallestDistanceSquared = distanceSquared;
            bestPointOnTriangle = closestPoint;
            bestVertexIndex = iVertex;
        }
    }
    bestPointOnTriangle = Util::mvmul(negRot, bestPointOnTriangle) + triangle[0];

    return {bestPointOnTriangle, smallestDistanceSquared, bestVertexIndex};
}

} // namespace CollisionSim::Util
