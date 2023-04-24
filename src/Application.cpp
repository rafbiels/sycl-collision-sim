/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "Application.h"
#include "Actor.h"
#include "CollisionCalculator.h"
#include "Constants.h"
#include "Shape.h"
#include "Util.h"
#include "World.h"

#include <Magnum/Magnum.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Math/Color.h>

#include <Corrade/Utility/FormatStl.h>
#include <Corrade/Containers/StringStlView.h>

// -----------------------------------------------------------------------------
CollisionSim::Application::Application(const Arguments& arguments)
: Magnum::Platform::Application{
    arguments,
    Configuration{}.setTitle(Constants::ApplicationName),
    GLConfiguration{}.setFlags(GLConfiguration::Flag::QuietLog)},
m_phongShader{Magnum::Shaders::PhongGL::Configuration{}.setLightCount(2)},
m_world{Magnum::Vector2{windowSize()}.aspectRatio(), Constants::DefaultWorldDimensions},
m_renderFrameTimeSec{Constants::FrameTimeCounterWindow},
m_computeFrameTimeSec{Constants::FrameTimeCounterWindow},
m_computeTask{[this]{compute();}},
m_syclQueue{std::make_unique<sycl::queue>(sycl::gpu_selector_v, sycl::property::queue::in_order{})}
{
    Magnum::GL::Renderer::enable(Magnum::GL::Renderer::Feature::DepthTest);
    Magnum::GL::Renderer::enable(Magnum::GL::Renderer::Feature::FaceCulling);
    Magnum::GL::Renderer::enable(Magnum::GL::Renderer::Feature::Blending);
    Magnum::GL::Renderer::setBlendFunction(Magnum::GL::Renderer::BlendFunction::One, Magnum::GL::Renderer::BlendFunction::OneMinusSourceAlpha);
    Magnum::GL::Renderer::setBlendEquation(Magnum::GL::Renderer::BlendEquation::Add, Magnum::GL::Renderer::BlendEquation::Add);

    setMinimalLoopPeriod(0);
    if (swapInterval()!=0) {setSwapInterval(0);}

    for (int iArg{0}; iArg<arguments.argc; ++iArg) {
        if (std::string{arguments.argv[iArg]}=="--cpu") {
            m_cpuOnly = true;
            Corrade::Utility::Debug{} << "Running CPU implementation without SYCL because of the --cpu option";
        }
    }
    if (!m_cpuOnly) {
        Corrade::Utility::Debug{} << "Running SYCL code on " << m_syclQueue->get_device().get_info<sycl::info::device::name>().c_str();
    }

    createActors();
    m_state = std::make_unique<State>(m_world.boundaries(), m_actors, m_numAllVertices);

    // Kernel warm-up
    CollisionCalculator::collideWorldParallel(m_syclQueue.get(), m_actors, m_state.get());

    m_textRenderer.newText("cfps",
        Magnum::Matrix3::projection(Magnum::Vector2{windowSize()})*
        Magnum::Matrix3::translation(Magnum::Vector2{windowSize()}*0.5f));
    m_textRenderer.newText("rfps",
        Magnum::Matrix3::projection(Magnum::Vector2{windowSize()})*
        Magnum::Matrix3::translation(Magnum::Vector2{windowSize()}*0.5f*Magnum::Vector2{1.0f,0.9f}));
    m_textRenderer.newText("clock",
        Magnum::Matrix3::projection(Magnum::Vector2{windowSize()})*
        Magnum::Matrix3::translation(Magnum::Vector2{windowSize()}*0.5f*Magnum::Vector2{1.0f,0.8f}));

    m_renderFrameTimer.reset();
    m_computeFrameTimer.reset();
    m_textUpdateTimer.reset();
    m_wallClock.reset();
    m_computeTask.start(Constants::ComputeInterval);
}

// -----------------------------------------------------------------------------
void CollisionSim::Application::tickEvent() {
    using namespace Magnum::Math::Literals;
    using FloatSecond = std::chrono::duration<float,std::ratio<1>>;

    float frameTimeSec{std::chrono::duration_cast<FloatSecond>(m_renderFrameTimer.step()).count()};
    m_renderFrameTimeSec.add(frameTimeSec);

    float wallTimeSec{std::chrono::duration_cast<FloatSecond>(m_wallClock.peek()).count() * Constants::RealTimeScale};

    if (m_textUpdateTimer.stepIfElapsed(Constants::TextUpdateInterval)) {
        float cfps{0.0f};
        {
            std::scoped_lock lock{m_computeFrameTimeSecMutex};
            cfps = 1.0f/m_computeFrameTimeSec.value();
            m_computeFrameTimeSec.reset();
        }
        m_textRenderer.get("cfps").renderer().render(Corrade::Utility::formatString("Compute FPS: {:.1f}", cfps));
        m_textRenderer.get("rfps").renderer().render(Corrade::Utility::formatString("Render FPS: {:.1f}", 1.0/m_renderFrameTimeSec.value()));
        m_textRenderer.get("clock").renderer().render(Corrade::Utility::formatString("Time: {:.1f}s", wallTimeSec));
        m_renderFrameTimeSec.reset();
    }
}

// -----------------------------------------------------------------------------
void CollisionSim::Application::drawEvent() {
    Magnum::GL::defaultFramebuffer.clear(
        Magnum::GL::FramebufferClear::Color |
        Magnum::GL::FramebufferClear::Depth);

    constexpr static Magnum::Color3 lightColour{250.0f/255.0f, 245.0f/255.0f, 240.0f/255.0f};

    m_phongShader.setLightPositions({{0.5, 1.0, 1.5, 0.0},{-0.5, 1.0, 1.5, 0.0}})
        .setLightColors({0.8f*lightColour, 0.2f*lightColour})
        .setSpecularColor(0.1f*lightColour)
        .setShininess(100.0f)
        .setProjectionMatrix(m_world.projection());
    for (Actor& actor : m_actors) {
        m_phongShader.setAmbientColor(0.8f*actor.colour())
            .setDiffuseColor(actor.colour())
            .setSpecularColor(0.1f*actor.colour())
            .setTransformationMatrix(actor.transformation())
            .setNormalMatrix(actor.transformation().normalMatrix())
            .draw(actor.mesh());
    }
    for (Shape& wall : m_world.walls()) {
        m_phongShader.setAmbientColor(0.8f*wall.colour())
            .setDiffuseColor(wall.colour())
            .setSpecularColor(0.1f*wall.colour())
            .setTransformationMatrix(wall.transformation())
            .setNormalMatrix(wall.transformation().normalMatrix())
            .draw(wall.mesh());
    }

    m_textRenderer.draw();

    redraw();
    swapBuffers();
}

// -----------------------------------------------------------------------------
void CollisionSim::Application::compute() {
    using namespace Magnum::Math::Literals;
    using FloatSecond = std::chrono::duration<float,std::ratio<1>>;
    float frameTimeSec{std::chrono::duration_cast<FloatSecond>(m_computeFrameTimer.step()).count()};
    {
        std::scoped_lock lock{m_computeFrameTimeSecMutex};
        m_computeFrameTimeSec.add(frameTimeSec);
    }
    float wallTimeSec{std::chrono::duration_cast<FloatSecond>(m_wallClock.peek()).count() * Constants::RealTimeScale};

    // Process world collision
    // std::cout << std::flush;
    // Corrade::Utility::Debug{} << "--------------------";
    // size_t iActor{0};
    // for (Actor& actor : m_actors) {
    //     Corrade::Utility::Debug{} << "actor " << iActor++ << " pos =" << actor.transformation().translation() << " v =" << actor.linearVelocity();
    // }
    // std::cout << std::flush;
    if (m_cpuOnly) {
        CollisionCalculator::collideWorldSequential(m_actors, m_world.boundaries());
    } else {
        CollisionCalculator::collideWorldParallel(m_syclQueue.get(), m_actors, m_state.get());
    }
    // std::cout << std::flush;
    // std::cout << std::flush;

    // Add global forces like gravity
    for (Actor& actor : m_actors) {
        // Add gravity
        actor.addForce({0.0f, m_world.gravity() * actor.mass(), 0.0f});
        // Add arbitrary extra force for testing the simulation
        if (wallTimeSec < 0.1) {
            actor.addForce({100.0f*actor.mass(), 0.0f, 0.0f}, {0.0f,0.0f,0.0f});
        }
        actor.computeState(Constants::RealTimeScale * frameTimeSec);
    }
}

// -----------------------------------------------------------------------------
void CollisionSim::Application::createActors() {
    using namespace Magnum::Math::Literals;

    constexpr static std::array<Magnum::Color3, 3> colours{
        Magnum::Color3{64.0f/255.0f, 106.0f/255.0f, 128.0f/255.0f},
        Magnum::Color3{65.0f/255.0f, 129.0f/255.0f, 97.0f/255.0f},
        Magnum::Color3{129.0f/255.0f, 65.0f/255.0f, 108.0f/255.0f}
    };

    const float xmin{m_world.boundaries().min().x()};
    const float xmax{m_world.boundaries().max().x()};
    const float zmin{m_world.boundaries().min().z()};
    const float zmax{m_world.boundaries().max().z()};
    const float xrange{xmax-xmin};
    const float zrange{zmax-zmin};
    const size_t gridSideN{5};
    const float dx{xrange/(gridSideN)};
    const float dz{zrange/(gridSideN)};
    auto generator = [](size_t index){
        size_t mod = index % 4;
        switch (mod) {
            case 0: return CollisionSim::ActorFactory::cube(0.7); break;
            case 1: return CollisionSim::ActorFactory::sphere(1.0); break;
            case 2: return CollisionSim::ActorFactory::cylinder(0.8); break;
            case 3: return CollisionSim::ActorFactory::cone(0.9); break;
            default: return CollisionSim::ActorFactory::cube(0.7); break;
        }
    };
    m_actors.reserve(gridSideN*gridSideN);
    for (size_t i{0}; i<gridSideN*gridSideN; ++i) {
        m_actors.emplace_back(generator(i));
        m_actors.back().transformation(
            Magnum::Matrix4::translation({xmin+dx*(0.5f+i/(gridSideN)),5.0,zmin+dz*(0.5f+i%(gridSideN))}) *
            Magnum::Matrix4::rotationX(20.0_degf*i) *
            Magnum::Matrix4::rotationY(15.0_degf*i)
        );
        m_actors.back().colour(colours[i % colours.size()]);
        m_numAllVertices += m_actors.back().numVertices();
    }
}

// -----------------------------------------------------------------------------
MAGNUM_APPLICATION_MAIN(CollisionSim::Application)
