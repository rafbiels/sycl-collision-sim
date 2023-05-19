/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_USMDATA
#define COLLISION_SIM_USMDATA

#include <sycl/sycl.hpp>
#include <vector>
#include <array>
#include <type_traits>

namespace CollisionSim {

namespace USMDataConcepts {
    template<size_t N>
    concept DynamicSize = N==0;
    template<size_t N>
    concept FixedSize = N>0;
}

/**
 * @struct USMData
 * @brief Memory object wrapper owning a data structure in host memory
 * and a device pointer to allocated memory with the same size.
 * 
 * Device allocation happens on construction and deallocation on destruction.
 * It also holds a non-owning pointer to sycl::queue to schedule the operations.
 * The lifetime of the queue must be ensured to be longer than the lifetime of
 * USMData, as the destructor needs to schedule the device memory deallocation.
 * 
 * The host data structure is std::vector if the size is dynamic (N=0) and
 * std::array if the size is fixed (N>0).
 */
template<typename T, size_t N=0>
struct USMData {
    USMData() = delete;
    /// Dynamic size constructor
    explicit USMData(size_t count, sycl::queue* q) requires USMDataConcepts::DynamicSize<N>
        : queue{q}, hostContainer(count), devicePointer{sycl::malloc_device<T>(count, *q)} {}
    /// Fixed size constructor
    explicit USMData(sycl::queue* q) requires USMDataConcepts::FixedSize<N>
        : queue{q}, devicePointer{sycl::malloc_device<T>(N, *q)} {}
    ~USMData() {
        sycl::free(devicePointer, *queue);
    }
    USMData(const USMData&) = delete;
    USMData(USMData&&) = delete;
    USMData& operator=(const USMData&) = delete;
    USMData& operator=(USMData&&) = delete;

    /// Immediate (no dependency) host-to-device memcpy; dynamic size version
    sycl::event copyToDevice() const requires USMDataConcepts::DynamicSize<N> {
        return queue->memcpy(devicePointer, hostContainer.data(), sizeof(T)*hostContainer.size());
    }
    /// Immediate (no dependency) host-to-device memcpy; fixed size version
    sycl::event copyToDevice() const requires USMDataConcepts::FixedSize<N> {
        return queue->memcpy(devicePointer, hostContainer.data(), sizeof(T)*N);
    }

    /// Event-dependent host-to-device memcpy; dynamic size version
    sycl::event copyToDevice(sycl::event depEvent) const requires USMDataConcepts::DynamicSize<N> {
        return queue->memcpy(devicePointer, hostContainer.data(), sizeof(T)*hostContainer.size(), depEvent);
    }
    /// Event-dependent host-to-device memcpy; fixed size version
    sycl::event copyToDevice(sycl::event depEvent) const requires USMDataConcepts::FixedSize<N> {
        return queue->memcpy(devicePointer, hostContainer.data(), sizeof(T)*N, depEvent);
    }

    /// Immediate (no dependency) device-to-host memcpy; dynamic size version
    sycl::event copyToHost() requires USMDataConcepts::DynamicSize<N> {
        return queue->memcpy(hostContainer.data(), devicePointer, sizeof(T)*hostContainer.size());
    }
    /// Immediate (no dependency) device-to-host memcpy; fixed size version
    sycl::event copyToHost() requires USMDataConcepts::FixedSize<N> {
        return queue->memcpy(hostContainer.data(), devicePointer, sizeof(T)*N);
    }

    /// Event-dependent device-to-host memcpy; dynamic size version
    sycl::event copyToHost(sycl::event depEvent) requires USMDataConcepts::DynamicSize<N> {
        return queue->memcpy(hostContainer.data(), devicePointer, sizeof(T)*hostContainer.size(), depEvent);
    }
    /// Event-dependent device-to-host memcpy; fixed size version
    sycl::event copyToHost(sycl::event depEvent) requires USMDataConcepts::FixedSize<N> {
        return queue->memcpy(hostContainer.data(), devicePointer, sizeof(T)*N, depEvent);
    }

    sycl::queue* queue; // non-owning pointer
    std::conditional_t<USMDataConcepts::DynamicSize<N>, std::vector<T>, std::array<T,N>> hostContainer{};
    T* devicePointer; // owning pointer
};

} // namespace CollisionSim

#endif // COLLISION_SIM_USMDATA
