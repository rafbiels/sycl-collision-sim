/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_APPLICATION
#define COLLISION_SIM_APPLICATION

#include "TextRenderer.h"
#include "Util.h"
#include "State.h"
#include "World.h"

#if HEADLESS
#include <Magnum/Platform/WindowlessGlxApplication.h>
#else
#include <Magnum/Platform/Sdl2Application.h>
#endif
#include <Magnum/Shaders/PhongGL.h>
#include <Corrade/Containers/Pointer.h>

#include <sycl/sycl.hpp>

#include <vector>
#include <mutex>

namespace CollisionSim {

class Actor;

#if HEADLESS
class Application : public Magnum::Platform::WindowlessGlxApplication
#else
class Application final : public Magnum::Platform::Application
#endif
{
    public:
        explicit Application(const Arguments& arguments);
        virtual ~Application();
        #if HEADLESS
        int exec() override;
        #endif

    private:
        #if !HEADLESS
        void tickEvent() override;
        void drawEvent() override;
        #else
        constexpr Magnum::Vector2i windowSize() {return {800, 600};}
        #endif
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
        Util::MovingAverage<float> m_computeFPSLongAvgSec;
        Util::MovingAverage<float> m_avgNumOverlaps;
        Util::MovingAverage<float> m_avgNumOverlapsLong;
        std::mutex m_computeFrameTimeSecMutex;
        /// Constant count of all vertices calculated at initialisation
        size_t m_numAllVertices{0};
        /// Constant count of all triangles calculated at initialisation
        size_t m_numAllTriangles{0};

        std::optional<SequentialState> m_sequentialState;
        std::optional<ParallelState> m_parallelState;
        std::optional<sycl::queue> m_syclQueue;

        bool m_cpuOnly{false};
};
} // namespace CollisionSim


#endif // COLLISION_SIM_APPLICATION
