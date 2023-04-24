/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_WORLD
#define COLLISION_SIM_WORLD

#include "Constants.h"
#include "Shape.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Range.h>
#include <array>

namespace CollisionSim {
class World {
    public:
        constexpr static size_t NumWalls{6};

        World(float windowAspectRatio,
              const Magnum::Vector3& dimensions,
              float gravity=Constants::EarthGravity);

        Magnum::Matrix4& projection();
        void projection(Magnum::Matrix4& proj);

        const Magnum::Range3D& boundaries() const;

        float gravity() const;

        std::array<Shape,NumWalls>& walls();

    private:
        Magnum::Matrix4 m_projection;
        Magnum::Range3D m_boundaries;
        float m_gravity{0};
        std::array<Shape,NumWalls> m_walls;
};
} // namespace CollisionSim

#endif // COLLISION_SIM_WORLD
