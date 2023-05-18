/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_UTIL
#define COLLISION_SIM_UTIL

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Vector4.h>
#include <Magnum/Math/Matrix3.h>
#include <Magnum/Math/Matrix4.h>
#include <sycl/sycl.hpp>
#include <chrono>
#include <deque>
#include <numeric>
#include <numbers>
#include <memory>
#include <thread>
#include <functional>
#include <utility>
#include <unordered_set>

namespace CollisionSim::Util {

class Timer {
    public:
        using clock_t = std::chrono::steady_clock;
        using time_point_t = clock_t::time_point;
        using duration_t = clock_t::duration;
        /**
         * Reset the stored time to now()
         */
        void reset();
        /**
         * Return the difference from now() to the previously stored
         * time without resetting the stored time
         */
        duration_t peek() const;
        /**
         * Reset the stored time to now() and return the difference
         * to the previously stored time
         */ 
        duration_t step();
        /**
         * Reset the stored time to now() if at least \c duration has
         * elapsed since the previously stored time
         */ 
        bool stepIfElapsed(duration_t duration);
    private:
        time_point_t m_currentTime;
};

class RepeatTask {
    public:
        RepeatTask(std::function<void()>&& callback);
        ~RepeatTask();
        RepeatTask(const RepeatTask&) = delete;
        RepeatTask(RepeatTask&&) = delete;
        RepeatTask& operator=(const RepeatTask&) = delete;
        RepeatTask& operator=(RepeatTask&&) = delete;
        void start(Timer::duration_t interval);
        void stop();
    private:
        void run();
        Timer m_timer;
        Timer::duration_t m_interval{0};
        std::unique_ptr<std::thread> m_thread;
        std::function<void()> m_callback;
        bool m_keepRunning{true};
};

template<typename T>
class MovingAverage {
    public:
        MovingAverage(size_t window) : m_window(window) {}
        T value() const {
            return std::accumulate(m_values.begin(), m_values.end(), T{0}) / static_cast<T>(m_values.size());
        }
        void add(T value) {
            if (m_values.size() >= m_window) {
                m_values.pop_front();
            }
            m_values.push_back(value);
        }
        void reset() {m_values.clear();};
    private:
        std::deque<T> m_values;
        size_t m_window{0};
};

struct ActorIndexPairHash {
    size_t operator()(std::pair<uint16_t,uint16_t> p) const {
        return (static_cast<size_t>(p.first) << 16) | static_cast<size_t>(p.second);
    }
};
class OverlapSet : public std::unordered_set<std::pair<uint16_t,uint16_t>,ActorIndexPairHash> {};


Magnum::Matrix3 outerProduct(const Magnum::Vector3& a, const Magnum::Vector3& b);

constexpr static float RoundingPrecision{1e6};
float round(float x);
Magnum::Vector3 round(const Magnum::Vector3& v);
Magnum::Vector4 round(const Magnum::Vector4& v);
Magnum::Matrix3 round(const Magnum::Matrix3& m);
Magnum::Matrix4 round(const Magnum::Matrix4& m);

void orthonormaliseRotation(Magnum::Matrix4& trfMatrix);

/// Magnum<->SYCL vector and matrix conversions
///@{
constexpr sycl::float3 toSycl(const Magnum::Vector3& vec) {
    const float (&data)[3] = vec.data();
    return sycl::float3{data[0],data[1],data[2]};
}
constexpr std::array<sycl::float3,3> toSycl(const Magnum::Matrix3& mat) {
    const float (&data)[9] = mat.data();
    return {sycl::float3{data[0],data[1],data[2]},
            sycl::float3{data[3],data[4],data[5]},
            sycl::float3{data[6],data[7],data[8]}};
}
constexpr Magnum::Vector3 toMagnum(const sycl::float3& vec) {
    return Magnum::Vector3{vec[0],vec[1],vec[2]};
}
constexpr Magnum::Matrix3 toMagnum(const std::array<sycl::float3,3>& mat) {
    return Magnum::Matrix3{
        Magnum::Vector3{mat[0][0], mat[0][1], mat[0][2]},
        Magnum::Vector3{mat[1][0], mat[1][1], mat[1][2]},
        Magnum::Vector3{mat[2][0], mat[2][1], mat[2][2]},
    };
}
constexpr Magnum::Matrix4 transformationMatrix(const sycl::float3& translation,
                                               const std::array<sycl::float3,3>& rotation) {
    return Magnum::Matrix4{
        Magnum::Vector4{rotation[0][0], rotation[0][1], rotation[0][2], 0.0f},
        Magnum::Vector4{rotation[1][0], rotation[1][1], rotation[1][2], 0.0f},
        Magnum::Vector4{rotation[2][0], rotation[2][1], rotation[2][2], 0.0f},
        Magnum::Vector4{translation[0], translation[1], translation[2], 1.0f}
    };
}
///@}

/// SYCL float3 equality operation
constexpr bool equal(const sycl::float3& a, const sycl::float3& b) {
    return a[0]==b[0] && a[1]==b[1] && a[2]==b[2];
}

/// SYCL matrix-scalar multiplication
constexpr std::array<sycl::float3,3> msmul(const std::array<sycl::float3,3>& mat, float scalar) {
    return {
        sycl::float3{mat[0][0], mat[0][1], mat[0][2]}*scalar,
        sycl::float3{mat[1][0], mat[1][1], mat[1][2]}*scalar,
        sycl::float3{mat[2][0], mat[2][1], mat[2][2]}*scalar
    };
}

/// SYCL matrix-vector multiplication
constexpr sycl::float3 mvmul(const std::array<sycl::float3,3>& mat, const sycl::float3& vec) {
    return {mat[0][0]*vec[0] + mat[1][0]*vec[1] + mat[2][0]*vec[2],
            mat[0][1]*vec[0] + mat[1][1]*vec[1] + mat[2][1]*vec[2],
            mat[0][2]*vec[0] + mat[1][2]*vec[1] + mat[2][2]*vec[2]};
}

/// SYCL matrix multiplication
constexpr std::array<sycl::float3,3> mmul(const std::array<sycl::float3,3>& a, const std::array<sycl::float3,3>& b) {
    return {
        sycl::float3{
            a[0][0]*b[0][0] + a[1][0]*b[0][1] + a[2][0]*b[0][2],
            a[0][1]*b[0][0] + a[1][1]*b[0][1] + a[2][1]*b[0][2],
            a[0][2]*b[0][0] + a[1][2]*b[0][1] + a[2][2]*b[0][2]
        },
        sycl::float3{
            a[0][0]*b[1][0] + a[1][0]*b[1][1] + a[2][0]*b[1][2],
            a[0][1]*b[1][0] + a[1][1]*b[1][1] + a[2][1]*b[1][2],
            a[0][2]*b[1][0] + a[1][2]*b[1][1] + a[2][2]*b[1][2]
        },
        sycl::float3{
            a[0][0]*b[2][0] + a[1][0]*b[2][1] + a[2][0]*b[2][2],
            a[0][1]*b[2][0] + a[1][1]*b[2][1] + a[2][1]*b[2][2],
            a[0][2]*b[2][0] + a[1][2]*b[2][1] + a[2][2]*b[2][2]
        }
    };
}

/// SYCL matrix transpose
constexpr std::array<sycl::float3,3> transpose(const std::array<sycl::float3,3>& mat) {
    return {
        sycl::float3{mat[0][0], mat[1][0], mat[2][0]},
        sycl::float3{mat[0][1], mat[1][1], mat[2][1]},
        sycl::float3{mat[0][2], mat[1][2], mat[2][2]}
    };
}

/// SYCL 3x3 matrix inverse
constexpr std::array<sycl::float3,3> inverse(const std::array<sycl::float3,3>& mat) {
    float a{mat[0][0]};
    float b{mat[1][0]};
    float c{mat[2][0]};
    float d{mat[0][1]};
    float e{mat[1][1]};
    float f{mat[2][1]};
    float g{mat[0][2]};
    float h{mat[1][2]};
    float i{mat[2][2]};
    float A{e*i - f*h};
    float B{f*g - d*i};
    float C{d*h - e*g};
    float D{c*h - b*i};
    float E{a*i - c*g};
    float F{b*g - a*h};
    float G{b*f - c*e};
    float H{c*d - a*f};
    float I{a*e - b*d};
    float detInv{1.0f/(a*A + b*B + c*C)};
    return {
        detInv * sycl::float3{A, B, C},
        detInv * sycl::float3{D, E, F},
        detInv * sycl::float3{G, H, I}
    };
}

/// @brief Returns the closest point P and distance d as {Px, Py, Pz, d} 
inline sycl::float4 closestPointOnTriangle(const std::array<sycl::float3,3>& triangle, const sycl::float3& point) {
    // ===========================================
    // "2D Method" following M.W. Jones 1995
    // 3D Distance from a Point to a Triangle
    // http://www-compsci.swan.ac.uk/~csmark/PDFS/1995_3D_distance_point_to_triangle
    // ===========================================
    // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> BEFORE: "
    //     << toMagnum(triangle[0]) << toMagnum(triangle[1]) << toMagnum(triangle[2]) << toMagnum(point);
    // Translate the problem to origin
    sycl::float3 A{0.0f};
    sycl::float3 B{triangle[1]-triangle[0]};
    sycl::float3 C{triangle[2]-triangle[0]};
    sycl::float3 P{point-triangle[0]};

    // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> TO ORIGIN: "
    //     << toMagnum(A) << toMagnum(B) << toMagnum(C) << toMagnum(P);

    // Rotate the problem around x-axis such that B lies in the xz plane
    const float denomRotX = sycl::sqrt(B[1]*B[1]+B[2]*B[2]);
    const float sinRotX = denomRotX==0 ? 0.0f : std::copysign(B[1] / denomRotX, B[2]);
    const float cosRotX = denomRotX==0 ? 1.0f : std::copysign(B[2] / denomRotX, B[1]);
    const std::array<sycl::float3,3> rotX = {
        sycl::float3{1.0f, 0.0f, 0.0f},
        sycl::float3{0.0f, cosRotX, sinRotX},
        sycl::float3{0.0f, -sinRotX, cosRotX}
    };
    B = mvmul(rotX, B);
    C = mvmul(rotX, C);
    P = mvmul(rotX, P);

    constexpr static float RadToDeg{180.0f/std::numbers::pi_v<float>};

    // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> ROT X: "
    //     << "sin = " << sinRotX << " cos = " << cosRotX << " sin^2+cos^2 = " << sinRotX*sinRotX+cosRotX*cosRotX
    //     << "asin = " << sycl::asin(sinRotX)*RadToDeg << "acos = " << sycl::acos(cosRotX)*RadToDeg;

    // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> ROT X: "
    //     << toMagnum(A) << toMagnum(B) << toMagnum(C) << toMagnum(P);

    // Rotate the problem around y-axis such that B lies on the z-axis
    const float denomRotY = sycl::sqrt(B[0]*B[0]+B[2]*B[2]);
    const float sinRotY = denomRotY==0 ? 0.0f : std::copysign(B[0] / denomRotY, B[2]);
    const float cosRotY = denomRotY==0 ? 1.0f : std::copysign(B[2] / denomRotY, -B[0]);
    const std::array<sycl::float3,3> rotY = {
        sycl::float3{cosRotY, 0.0f, -sinRotY},
        sycl::float3{0.0f, 1.0f, 0.0f},
        sycl::float3{sinRotY, 0.0f, cosRotY}
    };
    B = mvmul(rotY, B);
    C = mvmul(rotY, C);
    P = mvmul(rotY, P);

    // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> ROT Y: "
    //     << "sin = " << sinRotY << " cos = " << cosRotY << " sin^2+cos^2 = " << sinRotY*sinRotY+cosRotY*cosRotY
    //     << "asin = " << sycl::asin(sinRotY)*RadToDeg << "acos = " << sycl::acos(cosRotY)*RadToDeg;

    // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> ROT Y: "
    //     << toMagnum(A) << toMagnum(B) << toMagnum(C) << toMagnum(P);

    // Rotate the problem around z-axis such that C lies in the yz plane
    const float denomRotZ = sycl::sqrt(C[0]*C[0]+C[1]*C[1]);
    const float sinRotZ = denomRotZ==0 ? 0.0f : std::copysign(C[0] / denomRotZ, C[0]);
    const float cosRotZ = denomRotZ==0 ? 1.0f : std::copysign(C[1] / denomRotZ, -C[1]);
    const std::array<sycl::float3,3> rotZ = {
        sycl::float3{cosRotZ, -sinRotZ, 0.0f},
        sycl::float3{sinRotZ, cosRotZ, 0.0f},
        sycl::float3{0.0f, 0.0f, 1.0f}
    };
    B = mvmul(rotZ, B);
    C = mvmul(rotZ, C);
    P = mvmul(rotZ, P);

    // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> ROT Z: "
    //     << "sin = " << sinRotZ << " cos = " << cosRotZ << " sin^2+cos^2 = " << sinRotZ*sinRotZ+cosRotZ*cosRotZ
    //     << "asin = " << sycl::asin(sinRotZ)*RadToDeg << "acos = " << sycl::acos(cosRotZ)*RadToDeg;

    // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> ROT Z: "
    //     << toMagnum(A) << toMagnum(B) << toMagnum(C) << toMagnum(P);

    auto edge = [](float checkPointY, float checkPointZ,
                   float linePointY, float linePointZ,
                   float lineDY, float lineDZ) constexpr -> float {
        return (checkPointY-linePointY)*lineDZ - (checkPointZ-linePointZ)*lineDY;
    };

    float edgeAB = edge(P[1], P[2], A[1], A[2], B[1]-A[1], B[2]-A[2]);
    float edgeBC = edge(P[1], P[2], B[1], B[2], C[1]-B[1], C[2]-B[2]);
    float edgeCA = edge(P[1], P[2], C[1], C[2], A[1]-C[1], A[2]-C[2]);


    // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> 2D problem: "
    //     << "A=" << Magnum::Vector2{A[1],A[2]}
    //     << "B=" << Magnum::Vector2{B[1],B[2]}
    //     << "C=" << Magnum::Vector2{C[1],C[2]}
    //     << "P=" << Magnum::Vector2{P[1],P[2]};
    // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> edgeAB =" << edgeAB
    //     << " edgeBC =" << edgeBC
    //     << " edgeCA =" << edgeCA;

    float distance{std::numeric_limits<float>::lowest()};
    sycl::float3 closestPoint{0.0f, 0.0f, 0.0f};
    float clockwiseDet = -B[2]*C[1];
    // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> "
    //     << "clockwiseDet = " << clockwiseDet;

    if (clockwiseDet < 0) {
        // translate clockwise problem into anti-clockwise
        edgeAB = -edgeAB;
        edgeBC = -edgeBC;
        edgeCA = -edgeCA;
    }

    if (edgeAB <= 0 && edgeBC <= 0 && edgeCA <= 0) {
        // - - -
        // 2D point inside triangle
        // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> "
        //     << "2D point inside triangle";
        distance = sycl::abs(P[0]);
        closestPoint = sycl::float3{0.0f, P[1], P[2]};
    } else if (edgeAB >= 0) {
        if (edgeCA >= 0) {
            // + - + (edgeBC assumed -)
            // 2D point closest to triangle vertex A
            // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> "
            //     << "2D point closest to triangle vertex A";
            if (edgeBC > 0) {throw std::runtime_error("edgeBA expected <= 0 but is > 0");}
            distance = sycl::length(P);
            closestPoint = A;
        } else if (edgeBC >= 0) {
            // + + -
            // 2D point closest to triangle vertex B
            // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> "
            //     << "2D point closest to triangle vertex B";
            distance = sycl::sqrt(P[0]*P[0] + P[1]*P[1] + (P[2]-B[2])*(P[2]-B[2]));
            closestPoint = B;
        } else {
            // + - -
            // 2D point closest to triangle edge AB
            // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> "
            //     << "2D point closest to triangle edge AB";
            // AB lies on z-axis, so d(P, AB) is  d(P, z-axis)
            const float distanceYZSquared = P[1]*P[1];
            distance = sycl::sqrt(P[0]*P[0] + distanceYZSquared);
            const float ADOverAB = sycl::sqrt(
                ((P[1]-A[1])*(P[1]-A[1]) + (P[2]-A[2])*(P[2]-A[2]) - distanceYZSquared) /
                ((B[1]-A[1])*(B[1]-A[1]) + (B[2]-A[2])*(B[2]-A[2]))
            );
            closestPoint = sycl::float3{
                0.0f,
                A[1] + (B[1]-A[1])*ADOverAB,
                A[2] + (B[2]-A[2])*ADOverAB,
            };
        }
    } else if (edgeBC >= 0) {
        if (edgeCA >= 0) {
            // - + +
            // 2D point closest to triangle vertex C
            // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> "
            //     << "2D point closest to triangle vertex C";
            distance = sycl::sqrt(P[0]*P[0] + (P[1]-C[1])*(P[1]-C[1]) + (P[2]-C[2])*(P[2]-C[2]));
            closestPoint = C;
        } else {
            // - + -
            // 2D point closest to triangle edge BC
            // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> "
            //     << "2D point closest to triangle edge BC";
            const float BmPz = B[2]-P[2];
            const float CmBz = C[2]-B[2];
            const float distanceYZSquared = (C[1]*BmPz + P[1]*CmBz) * (C[1]*BmPz + P[1]*CmBz) / (C[1]*C[1] + CmBz*CmBz);
            distance = sycl::sqrt(P[0]*P[0] + distanceYZSquared);
            const float BDOverBC = sycl::sqrt(
                ((P[1]-B[1])*(P[1]-B[1]) + (P[2]-B[2])*(P[2]-B[2]) - distanceYZSquared) /
                ((C[1]-B[1])*(C[1]-B[1]) + (C[2]-B[2])*(C[2]-B[2]))
            );
            closestPoint = sycl::float3{
                0.0f,
                B[1] + (C[1]-B[1])*BDOverBC,
                B[2] + (C[2]-B[2])*BDOverBC,
            };
        }
    } else {
        // - - + (edgeCA assumed +)
        // 2D point closest to triangle edge CA
        // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> "
        //     << "2D point closest to triangle edge CA";
        if (edgeCA < 0) {throw std::runtime_error("edgeCA expected >= 0 but is < 0");}
        const float distanceYZSquared = (C[2]*P[1]-C[1]*P[2]) * (C[2]*P[1]-C[1]*P[2]) / (C[1]*C[1]+C[2]*C[2]);
        distance = sycl::sqrt(P[0]*P[0] + distanceYZSquared);
        const float CDOverCA = sycl::sqrt(
            ((P[1]-C[1])*(P[1]-C[1]) + (P[2]-C[2])*(P[2]-C[2]) - distanceYZSquared) /
            ((A[1]-C[1])*(A[1]-C[1]) + (A[2]-C[2])*(A[2]-C[2]))
        );
        closestPoint = sycl::float3{
            0.0f,
            C[1] + (A[1]-C[1])*CDOverCA,
            C[2] + (A[2]-C[2])*CDOverCA,
        };
    }

    // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> "
    //     << "distance = " << distance
    //     << ", closestPoint(transformed) = " << Util::toMagnum(closestPoint);

    const std::array<sycl::float3,3> negRotX = {
        sycl::float3{1.0f, 0.0f, 0.0f},
        sycl::float3{0.0f, cosRotX, -sinRotX},
        sycl::float3{0.0f, sinRotX, cosRotX}
    };
    const std::array<sycl::float3,3> negRotY = {
        sycl::float3{cosRotY, 0.0f, sinRotY},
        sycl::float3{0.0f, 1.0f, 0.0f},
        sycl::float3{-sinRotY, 0.0f, cosRotY}
    };
    const std::array<sycl::float3,3> negRotZ = {
        sycl::float3{cosRotZ, sinRotZ, 0.0f},
        sycl::float3{-sinRotZ, cosRotZ, 0.0f},
        sycl::float3{0.0f, 0.0f, 1.0f}
    };

    // Transform back to original coordinate system
    closestPoint = Util::mvmul(negRotX, Util::mvmul(negRotY, Util::mvmul(negRotZ, closestPoint))) + triangle[0];

    // Corrade::Utility::Debug{} << "closestPointOnTriangle\n>>> "
    //     << "closestPoint(real coords) = " << Util::toMagnum(closestPoint);

    return sycl::float4{closestPoint[0], closestPoint[1], closestPoint[2], distance};
}

} // namespace CollisionSim::Util

#endif // COLLISION_SIM_UTIL
