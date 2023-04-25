/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_STATE
#define COLLISION_SIM_STATE

#include "Actor.h"
#include "Wall.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Range.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <array>

namespace CollisionSim {

enum class StateVariable : uint8_t {
    ActorIndices=0, Mass, BodyInertiaInv, BodyVertexX, BodyVertexY, BodyVertexZ,
    VertexX, VertexY, VertexZ, Translation, Rotation, InertiaInv,
    LinearVelocity, AngularVelocity, LinearMomentum, AngularMomentum,
    Force, Torque, WallCollisions, WorldBoundaries,
    MAX
};

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
        Corrade::Utility::Debug{} << "Enqueuing h2d copy from " << hostContainer.data() << " to " << devicePointer << "(" << hostContainer.size() << " elements)";
        return queue->memcpy(devicePointer, hostContainer.data(), sizeof(T)*hostContainer.size());
    }
    sycl::event copyToDevice(sycl::event depEvent) const {
        Corrade::Utility::Debug{} << "Enqueuing h2d copy from " << hostContainer.data() << " to " << devicePointer << "(" << hostContainer.size() << " elements)";
        return queue->memcpy(devicePointer, hostContainer.data(), sizeof(T)*hostContainer.size(), depEvent);
    }
    sycl::event copyToHost() {
        Corrade::Utility::Debug{} << "Enqueuing d2h copy from " << devicePointer << " to " << hostContainer.data() << "(" << hostContainer.size() << " elements)";
        return queue->memcpy(hostContainer.data(), devicePointer, sizeof(T)*hostContainer.size());
    }
    sycl::event copyToHost(sycl::event depEvent) {
        Corrade::Utility::Debug{} << "Enqueuing d2h copy from " << devicePointer << " to " << hostContainer.data() << "(" << hostContainer.size() << " elements)";
        return queue->memcpy(hostContainer.data(), devicePointer, sizeof(T)*hostContainer.size(), depEvent);
    }

    sycl::queue* queue; // non-owning pointer
    std::vector<T> hostContainer;
    T* devicePointer; // owning pointer
};

/**
 * Class representing the simulation state, with properties
 * of all actors formatted into contiguous arrays
 */
class State {
    public:
        using float3x3 = std::array<sycl::float3,3>;

        /// Empty constructor
        State() = delete;
        /**
         * Constructor from a vector of actors
         */
        State(const Magnum::Range3D& worldBoundaries,
              const std::vector<Actor>& actors,
              size_t numAllVertices,
              sycl::queue* queue);

        /// Copy and assignment explicitly deleted
        ///@{
        State(const State&) = delete;
        State(State&&) = delete;
        State& operator=(const State&) = delete;
        State& operator=(State&&) = delete;
        ///}

        /// Enqueue copy of all data to the device and return immediately
        void copyAllToDeviceAsync() const;

        size_t numActors{0};
        size_t numAllVertices{0};

        USMData<float> worldBoundaries;
        USMData<uint16_t> actorIndices; // Caution: restricting numActors to 65536
        USMData<float> mass;
        USMData<float3x3> bodyInertiaInv;
        std::array<USMData<float>,3> bodyVertices;
        std::array<USMData<float>,3> worldVertices;
        USMData<sycl::float3> translation;
        USMData<float3x3> rotation;
        USMData<float3x3> inertiaInv;
        USMData<sycl::float3> linearVelocity;
        USMData<sycl::float3> angularVelocity;
        USMData<sycl::float3> linearMomentum;
        USMData<sycl::float3> angularMomentum;
        USMData<sycl::float3> force;
        USMData<sycl::float3> torque;
        USMData<Wall> wallCollisions;
};
} // namespace CollisionSim

#endif // COLLISION_SIM_STATE
