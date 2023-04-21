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
#include "State.h"
#include "World.h"

#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Shaders/PhongGL.h>
#include <Corrade/Containers/Pointer.h>

#include <sycl/sycl.hpp>

#include <vector>
#include <mutex>

namespace CollisionSim {
class Application final : public Magnum::Platform::Application {
    public:
        explicit Application(const Arguments& arguments);
        virtual ~Application() {m_computeTask.stop();}

    private:
        void tickEvent() override;
        void drawEvent() override;
        void compute();
        void createActors();

        Magnum::Shaders::PhongGL m_phongShader;

        World m_world;
        std::vector<Actor> m_actors;

        Util::Timer m_renderFrameTimer;
        Util::Timer m_computeFrameTimer;
        Util::Timer m_textUpdateTimer;
        Util::Timer m_wallClock;
        Util::RepeatTask m_computeTask;
        TextRenderer m_textRenderer;
        Util::MovingAverage<float> m_renderFrameTimeSec;
        Util::MovingAverage<float> m_computeFrameTimeSec;
        std::mutex m_computeFrameTimeSecMutex;
        /// Constant count of all vertices calculated at initialisation
        size_t m_numAllVertices{0};

        std::unique_ptr<State> m_state;
        std::unique_ptr<sycl::queue> m_syclQueue;
};
} // namespace CollisionSim


#endif // COLLISION_SIM_APPLICATION
