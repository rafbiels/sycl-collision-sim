/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_WORLD
#define COLLISION_SIM_WORLD

#include <Magnum/Magnum.h>
#include <Magnum/Math/Matrix4.h>

namespace CollisionSim {
class World {
    public:
        World(float windowAspectRatio);

        Magnum::Matrix4& projection();
        void projection(Magnum::Matrix4& proj);

    private:
        Magnum::Matrix4 m_projection;
};
} // namespace CollisionSim

#endif // COLLISION_SIM_WORLD
