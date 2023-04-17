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
: Magnum::Platform::Application{arguments, Configuration{}.setTitle(Constants::ApplicationName)},
m_phongShader{Magnum::Shaders::PhongGL::Configuration{}.setLightCount(2)},
m_world{Magnum::Vector2{windowSize()}.aspectRatio(), Constants::DefaultWorldDimensions},
m_renderFrameTimeSec{Constants::FrameTimeCounterWindow},
m_computeFrameTimeSec{Constants::FrameTimeCounterWindow},
m_computeTask{[this]{compute();}}
{
    Magnum::GL::Renderer::enable(Magnum::GL::Renderer::Feature::DepthTest);
    Magnum::GL::Renderer::enable(Magnum::GL::Renderer::Feature::FaceCulling);
    Magnum::GL::Renderer::enable(Magnum::GL::Renderer::Feature::Blending);
    Magnum::GL::Renderer::setBlendFunction(Magnum::GL::Renderer::BlendFunction::One, Magnum::GL::Renderer::BlendFunction::OneMinusSourceAlpha);
    Magnum::GL::Renderer::setBlendEquation(Magnum::GL::Renderer::BlendEquation::Add, Magnum::GL::Renderer::BlendEquation::Add);

    setMinimalLoopPeriod(0);
    setSwapInterval(0);

    using namespace Magnum::Math::Literals;

    constexpr static Magnum::Color3 defaultColour{64.0f/255.0f, 106.0f/255.0f, 128.0f/255.0f};

    m_actors.reserve(4);
    // Add a cube
    Actor& cube = m_actors.emplace_back(
        CollisionSim::ActorFactory::cube(1.0));
    cube.transformation() = Magnum::Matrix4::translation({-6.0,5.0,0.0}) * Magnum::Matrix4::rotationX(30.0_degf);
    cube.colour(defaultColour);
    // Add a sphere
    Actor& sphere = m_actors.emplace_back(
        CollisionSim::ActorFactory::sphere(2.0, 2));
    sphere.transformation() = Magnum::Matrix4::translation({-2.0,5.0,0.0}) * Magnum::Matrix4::rotationX(30.0_degf);
    sphere.colour(defaultColour);
    // Add a cylinder
    Actor& cylinder = m_actors.emplace_back(
        CollisionSim::ActorFactory::cylinder(1.2, 4, 20, 1.0));
    cylinder.transformation() = Magnum::Matrix4::translation({2.0,5.0,0.0}) * Magnum::Matrix4::rotationX(30.0_degf);
    cylinder.colour(defaultColour);
    // Add a cone
    Actor& cone = m_actors.emplace_back(
        CollisionSim::ActorFactory::cone(1.0, 4, 20, 1.0));
    cone.transformation() = Magnum::Matrix4::translation({6.0,5.0,0.0}) * Magnum::Matrix4::rotationX(30.0_degf);
    cone.colour(defaultColour);

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
    CollisionCalculator::collideWorldSequential(m_actors, m_world.boundaries());

    // Add global forces like gravity
    for (Actor& actor : m_actors) {
        // Add gravity
        actor.addForce({0.0f, m_world.gravity() * actor.mass(), 0.0f});
        // Add arbitrary extra force for testing the simulation
        if (wallTimeSec < 0.1) {
            actor.addForce({0.0f, 0.0f, 100.0f*actor.mass()}, {0.0f,0.0f,0.0f});
        }
        actor.computeState(Constants::RealTimeScale * frameTimeSec);
    }
}

// -----------------------------------------------------------------------------
MAGNUM_APPLICATION_MAIN(CollisionSim::Application)
