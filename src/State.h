/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_STATE
#define COLLISION_SIM_STATE

#include "Actor.h"
#include "Constants.h"
#include "Util.h"
#include "Wall.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Range.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <array>
#include <utility>

namespace CollisionSim {

template<typename T>
struct USMData {
    USMData() = delete;
    explicit USMData(size_t count, sycl::queue* q)
        : queue{q}, hostContainer(count), devicePointer{sycl::malloc_device<T>(count, *q)} {}
    ~USMData() {
        sycl::free(devicePointer, *queue);
    }
    USMData(const USMData&) = delete;
    USMData(USMData&&) = delete;
    USMData& operator=(const USMData&) = delete;
    USMData& operator=(USMData&&) = delete;
    sycl::event copyToDevice() const {
        return queue->memcpy(devicePointer, hostContainer.data(), sizeof(T)*hostContainer.size());
    }
    sycl::event copyToDevice(sycl::event depEvent) const {
        return queue->memcpy(devicePointer, hostContainer.data(), sizeof(T)*hostContainer.size(), depEvent);
    }
    sycl::event copyToHost() {
        return queue->memcpy(hostContainer.data(), devicePointer, sizeof(T)*hostContainer.size());
    }
    sycl::event copyToHost(sycl::event depEvent) {
        return queue->memcpy(hostContainer.data(), devicePointer, sizeof(T)*hostContainer.size(), depEvent);
    }

    sycl::queue* queue; // non-owning pointer
    std::vector<T> hostContainer;
    T* devicePointer; // owning pointer
};

struct Edge {
    uint16_t actorIndex{std::numeric_limits<uint16_t>::max()};
    bool isEnd{false};
};

template <size_t... Indices>
consteval std::array<Edge, 2*sizeof...(Indices)> edgeArray(std::index_sequence<Indices...>) {
    return {{(Edge{Indices, false})..., (Edge{Indices, true})...}};
}

inline Corrade::Utility::Debug& operator<<(Corrade::Utility::Debug& s, Edge e) {
    return s << std::to_string(e.actorIndex).append(1, e.isEnd ? 'e' : 's').c_str();
}

/**
 * Class representing the simulation state for parallel simulation,
 * with properties of all actors formatted into contiguous arrays
 */
class ParallelState {
    public:
        using float3x3 = std::array<sycl::float3,3>;

        /// Empty constructor
        ParallelState() = delete;
        /**
         * Constructor from a vector of actors
         */
        explicit ParallelState(const Magnum::Range3D& worldBounds,
                               const std::vector<Actor>& actors,
                               size_t numAllVertices,
                               sycl::queue* queue);

        /// Copy and assignment explicitly deleted
        ///@{
        ParallelState(const ParallelState&) = delete;
        ParallelState(ParallelState&&) = delete;
        ParallelState& operator=(const ParallelState&) = delete;
        ParallelState& operator=(ParallelState&&) = delete;
        ///}

        /// Enqueue copy of all data to the device and return immediately
        void copyAllToDeviceAsync() const;

        size_t numActors{0};
        size_t numAllVertices{0};

        /// Constants
        ///@{
        USMData<float> worldBoundaries;
        USMData<uint16_t> actorIndices; // Caution: restricting numActors to 65536
        USMData<float> mass;
        USMData<float3x3> bodyInertiaInv;
        std::array<USMData<float>,3> bodyVertices;
        ///@}

        /// Motion simulation variables
        ///@{
        USMData<sycl::float3> translation;
        USMData<float3x3> rotation;
        USMData<float3x3> inertiaInv;
        USMData<sycl::float3> linearVelocity;
        USMData<sycl::float3> angularVelocity;
        USMData<sycl::float3> linearMomentum;
        USMData<sycl::float3> angularMomentum;
        USMData<sycl::float3> force;
        USMData<sycl::float3> torque;
        ///@}

        /// Collision simulation variables
        ///@{
        USMData<Wall> wallCollisions;
        USMData<sycl::float3> addLinearVelocity; /// Per-vertex collision response
        USMData<sycl::float3> addAngularVelocity; /// Per-vertex collision response
        ///@}
};

/**
 * Class holding static data for sequential simulation
 */
class SequentialState {
    public:
        SequentialState() = delete;
        explicit SequentialState(const Magnum::Range3D& worldBounds);

        /// Copy and assignment explicitly deleted
        ///@{
        SequentialState(const SequentialState&) = delete;
        SequentialState(SequentialState&&) = delete;
        SequentialState& operator=(const SequentialState&) = delete;
        SequentialState& operator=(SequentialState&&) = delete;
        ///}

        /// State variables
        ///@{
        Magnum::Range3D worldBoundaries;
        std::array<std::array<Edge, 2*Constants::NumActors>,3> sortedAABBEdges{
            edgeArray(std::make_index_sequence<Constants::NumActors>{}), // x
            edgeArray(std::make_index_sequence<Constants::NumActors>{}), // y
            edgeArray(std::make_index_sequence<Constants::NumActors>{}), // z
        };
        Util::OverlapSet aabbOverlaps;
        ///}
};

} // namespace CollisionSim

#endif // COLLISION_SIM_STATE
