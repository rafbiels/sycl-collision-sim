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
 *
 * The host data structure is std::vector if the size is dynamic (N=0) and
 * std::array if the size is fixed (N>0).
 */
template<typename T, size_t N=0>
struct USMData {
    USMData() = delete;
    /// Dynamic size constructor
    explicit USMData(const sycl::queue& q, size_t count) requires USMDataConcepts::DynamicSize<N>
        : queue{q}, hostContainer(count), devicePointer{sycl::malloc_device<T>(count, q)} {}
    /// Fixed size constructor
    explicit USMData(const sycl::queue& q, std::array<T,N>&& values={}) requires USMDataConcepts::FixedSize<N>
        : queue{q}, hostContainer{values}, devicePointer{sycl::malloc_device<T>(N, q)} {}
    ~USMData() {
        sycl::free(devicePointer, queue);
    }
    USMData(const USMData&) = delete;
    USMData(USMData&&) = delete;
    USMData& operator=(const USMData&) = delete;
    USMData& operator=(USMData&&) = delete;

    constexpr size_t size() const {
        if constexpr (USMDataConcepts::DynamicSize<N>) {
            return hostContainer.size();
        } else {
            return N;
        }
    }

    /// Host-to-device memcpy
    template<typename... Args>
    sycl::event copyToDevice(Args&&... args) const {
        return queue.memcpy(devicePointer, hostContainer.data(), sizeof(T)*size(), std::forward<Args>(args)...);
    }

    /// Device-to-host memcpy
    template<typename... Args>
    sycl::event copyToHost(Args&&... args) {
        return queue.memcpy(hostContainer.data(), devicePointer, sizeof(T)*size(), std::forward<Args>(args)...);
    }

    mutable sycl::queue queue; // queue to schedule memcpy
    std::conditional_t<USMDataConcepts::DynamicSize<N>, std::vector<T>, std::array<T,N>> hostContainer;
    T* devicePointer; // owning pointer
};

} // namespace CollisionSim

#endif // COLLISION_SIM_USMDATA
