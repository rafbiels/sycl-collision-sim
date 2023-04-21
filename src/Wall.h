/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_WALL
#define COLLISION_SIM_WALL

#include <sycl/sycl.hpp>

namespace CollisionSim {
using WallUnderlyingType = uint8_t;
enum class Wall : WallUnderlyingType {
    None = 0,
    Xmin = 1,
    Xmax = 1<<1,
    Ymin = 1<<2,
    Ymax = 1<<3,
    Zmin = 1<<4,
    Zmax = 1<<5
};
constexpr Wall& operator|=(Wall& a, WallUnderlyingType b) {
    a = static_cast<Wall>(static_cast<WallUnderlyingType>(a) | b);
    return a;
}
constexpr Wall& operator|=(Wall& a, Wall b) {
    a = static_cast<Wall>(static_cast<WallUnderlyingType>(a) | static_cast<WallUnderlyingType>(b));
    return a;
}
constexpr WallUnderlyingType operator&(Wall a, Wall b) {
    return static_cast<WallUnderlyingType>(a) & static_cast<WallUnderlyingType>(b);
}
constexpr sycl::float3 wallNormal(Wall wall) {
    return {static_cast<float>(-((wall & Wall::Xmax) >> 1) + ((wall & Wall::Xmin) >> 0)),
            static_cast<float>(-((wall & Wall::Ymax) >> 3) + ((wall & Wall::Ymin) >> 2)),
            static_cast<float>(-((wall & Wall::Zmax) >> 5) + ((wall & Wall::Zmin) >> 4))};
}
}

#endif // COLLISION_SIM_WALL
