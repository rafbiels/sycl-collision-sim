/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_CONSTANTS
#define COLLISION_SIM_CONSTANTS

#include "Util.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <chrono>
#include <string_view>

namespace CollisionSim::Constants {

/// Application name shown in the window title bar
constexpr static std::string_view ApplicationName{"Collision Simulation"};

/// Minimum amount of time between two calls to compute state of the simulation
constexpr static Util::Timer::duration_t ComputeInterval{std::chrono::microseconds{0}};

/// Minimum amount of time between two updates of the on-screen text (e.g. FPS counter)
constexpr static Util::Timer::duration_t TextUpdateInterval{std::chrono::milliseconds{200}};

/// Number of measurements to keep for averaging the FPS measurement
constexpr static size_t FrameTimeCounterWindow{2048};

/// Slow down or speed up the simulation with respect to real time
constexpr static float RealTimeScale{0.5f};

/// Simulation size given by the square root of the number of actors
constexpr static size_t SqrtNumActors{ACTOR_GRID_SIZE};

/// Number of actors
constexpr static size_t NumActors{SqrtNumActors*SqrtNumActors};

/// Maximum number of triangles in the mesh of a single actor (used to allocate arrays)
constexpr static size_t MaxNumTriangles{320};

/// Scaling factors with respect to SI units
namespace Units {
constexpr static float Distance{100.f}; // cm
constexpr static float Area{Distance*Distance}; // cm^2
constexpr static float Volume{Distance*Distance*Distance}; // cm^3
constexpr static float Mass{1.0f}; // kg
constexpr static float Time{1.0f}; // s
constexpr static float Density{Mass/Volume}; // kg/cm^3
constexpr static float Velocity{Distance/Time}; // cm/s
}

/// Default dimensions of the world
constexpr static Magnum::Vector3 DefaultWorldDimensions{10.0f,8.0f,10.0f};

/// Uniform density of the body materials
constexpr static float DefaultDensity{1000.0f * Units::Density}; // kg/cm^3, approx. water density

/// Gravity
constexpr static float EarthGravity{-9.81f * Units::Distance / (Units::Time * Units::Time)}; // m/s^2

/// Restitution coefficient (fraction of kinematic energy conserved in a collision)
constexpr static float RestitutionCoefficient{1.0f};

/// Narrow phase collision threshold for triangle-vertex distance between actors
constexpr static float NarrowPhaseCollisionThreshold{0.001f*0.001f};

/// Look-up table binomial coefficient of n^2 for small numbers
template<size_t n>
consteval size_t binomialCoefficientOfSquare() {
    static_assert(n<21);
    std::array<size_t,21> lut{{0, 0, 6, 36, 120, 300, 630, 1176, 2016, 3240,
                               4950, 7260, 10296, 14196, 19110, 25200, 32640,
                               41616, 52326, 64980, 79800}};
    return lut[n];
}

constexpr static size_t NumActorPairs{binomialCoefficientOfSquare<Constants::SqrtNumActors>()};

/// Compile-time list of non-repeat pairs among N=NumActors items
consteval std::array<std::pair<size_t,size_t>,NumActorPairs> _actorPairsGenerator() {
    std::array<std::pair<size_t,size_t>,NumActorPairs> ret;
    size_t linear_index{0};
    for (size_t i{0}; i<Constants::NumActors; ++i) {
        for (size_t j{i+1}; j<Constants::NumActors; ++j) {
            ret[linear_index] = {i,j};
            ++linear_index;
        }
    }
    return ret;
}

constexpr static std::array<std::pair<size_t,size_t>,NumActorPairs> ActorPairs{_actorPairsGenerator()};

} // namespace CollisionSim::ConstantUtil

#endif // COLLISION_SIM_CONSTANTS
