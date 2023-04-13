/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_WORLD
#define COLLISION_SIM_WORLD

#include "Constants.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Matrix4.h>

namespace CollisionSim {
class World {
    public:
        World(float windowAspectRatio,
              float gravity=Constants::EarthGravity);

        Magnum::Matrix4& projection();
        void projection(Magnum::Matrix4& proj);

        float gravity() const {return m_gravity;}
    private:
        Magnum::Matrix4 m_projection;
        float m_gravity{0};
};
} // namespace CollisionSim

#endif // COLLISION_SIM_WORLD
