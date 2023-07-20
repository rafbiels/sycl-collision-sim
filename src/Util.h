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
        size_t size() const {return m_values.size();}
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
template<typename T>
constexpr sycl::vec<T,3> toSycl(const Magnum::Math::Vector<3,T>& vec) {
    const T (&data)[3] = vec.data();
    return {data[0],data[1],data[2]};
}
constexpr std::array<sycl::float3,3> toSycl(const Magnum::Matrix3& mat) {
    const float (&data)[9] = mat.data();
    return {sycl::float3{data[0],data[1],data[2]},
            sycl::float3{data[3],data[4],data[5]},
            sycl::float3{data[6],data[7],data[8]}};
}
template<typename T>
constexpr Magnum::Math::Vector<3,T> toMagnum(const sycl::vec<T,3>& vec) {
    return {vec[0],vec[1],vec[2]};
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

/// Create a ND-range maximising the use of all Compute Units
/// (total N work items may be slightly larger than requested)
sycl::nd_range<1> ndRangeAllCU(size_t minNumWorkItems, const sycl::queue& queue);

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

/**
 * @brief Transform a triangle to a coordinate system where its first vertex is
 * at the origin, the second vertex lies on z-axis and the third in the yz-plane.
 *
 * @return 3 matrices 3x3: the transformed triangle (vertices in columns),
 * the rotation matrix to transform points to the new coordinate system,
 * the rotation matrix to transform points back to the original coordinate system
 *
 * Note the full transform also requires a translation which is a vector opposite
 * to the location of the first triangle vertex.
 */
SYCL_EXTERNAL extern
std::array<std::array<sycl::float3,3>,3> triangleTransform(const std::array<sycl::float3,3>& triangle);

/// Returns the closest point and distance squared d^2
/// assuming the coordinate system from \c triangleTransform
SYCL_EXTERNAL extern
std::pair<sycl::float3,float> closestPointOnTriangle(const std::array<sycl::float3,3>& triangle, const sycl::float3& point);

/// Structure returned by \c closestPointOnTriangleND
struct ClosestPointOnTriangleReturnValue {
    sycl::float3 bestPointOnTriangle{0.0f,0.0f,0.0f};
    float distanceSquared{0.0f};
    size_t iVertex{0};
};

/// Returns the closest point and distance squared across all \c vertices
ClosestPointOnTriangleReturnValue closestPointOnTriangleND(
    const std::array<sycl::float3,3>& triangle,
    const std::vector<Magnum::Vector3>& vertices);

} // namespace CollisionSim::Util

#endif // COLLISION_SIM_UTIL
