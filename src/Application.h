/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_APPLICATION
#define COLLISION_SIM_APPLICATION

#include "Actor.h"
#include "TextRenderer.h"
#include "Util.h"
#include "World.h"

#include <Corrade/Containers/Pointer.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Shaders/PhongGL.h>

#include <memory>
#include <vector>

namespace CollisionSim {
class Application final : public Magnum::Platform::Application {
    public:
        explicit Application(const Arguments& arguments);

    private:
        void tickEvent() override;
        void drawEvent() override;

        Magnum::Shaders::PhongGL m_phongShader;

        World m_world;
        std::vector<Actor> m_actors;

        Util::Timer m_frameTimer;
        Util::Timer m_textUpdateTimer;
        TextRenderer m_textRenderer;
        Util::MovingAverage<float> m_frameTimeSec;
};
} // namespace CollisionSim


#endif // COLLISION_SIM_APPLICATION
