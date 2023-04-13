/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "Application.h"
#include "Actor.h"
#include "Constants.h"
#include "Util.h"
#include "World.h"
#include <Corrade/Utility/FormatStl.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>

#include <Magnum/Magnum.h>

#include <Corrade/Containers/StringStlView.h>
#include <Corrade/Utility/Debug.h>
#include <chrono>
#include <memory>

#include <Magnum/Math/Color.h>

// -----------------------------------------------------------------------------
CollisionSim::Application::Application(const Arguments& arguments)
: Magnum::Platform::Application{arguments, Configuration{}.setTitle(Constants::ApplicationName)},
m_world{Magnum::Vector2{windowSize()}.aspectRatio()},
m_frameTimeSec{Constants::FrameTimeCounterWindow}
{
    Magnum::GL::Renderer::enable(Magnum::GL::Renderer::Feature::DepthTest);
    Magnum::GL::Renderer::enable(Magnum::GL::Renderer::Feature::FaceCulling);
    Magnum::GL::Renderer::enable(Magnum::GL::Renderer::Feature::Blending);
    Magnum::GL::Renderer::setBlendFunction(Magnum::GL::Renderer::BlendFunction::One, Magnum::GL::Renderer::BlendFunction::OneMinusSourceAlpha);
    Magnum::GL::Renderer::setBlendEquation(Magnum::GL::Renderer::BlendEquation::Add, Magnum::GL::Renderer::BlendEquation::Add);

    setMinimalLoopPeriod(0);
    setSwapInterval(0);

    using namespace Magnum::Math::Literals;

    // Add a cube
    Actor& cube = m_actors.emplace_back(
        CollisionSim::ActorFactory::cube(1.0));
    cube.transformation() = Magnum::Matrix4::translation({-6.0,5.0,0.0}) * Magnum::Matrix4::rotationX(30.0_degf);
    // Add a sphere
    Actor& sphere = m_actors.emplace_back(
        CollisionSim::ActorFactory::sphere(2.0, 2));
    sphere.transformation() = Magnum::Matrix4::translation({-2.0,5.0,0.0}) * Magnum::Matrix4::rotationX(30.0_degf);
    // Add a cylinder
    Actor& cylinder = m_actors.emplace_back(
        CollisionSim::ActorFactory::cylinder(1.2, 4, 20, 1.0));
    cylinder.transformation() = Magnum::Matrix4::translation({2.0,5.0,0.0}) * Magnum::Matrix4::rotationX(30.0_degf);
    // Add a cone
    Actor& cone = m_actors.emplace_back(
        CollisionSim::ActorFactory::cone(1.0, 4, 20, 1.0));
    cone.transformation() = Magnum::Matrix4::translation({6.0,5.0,0.0}) * Magnum::Matrix4::rotationX(30.0_degf);

    m_textRenderer.newText("fps",
        Magnum::Matrix3::projection(Magnum::Vector2{windowSize()})*
        Magnum::Matrix3::translation(Magnum::Vector2{windowSize()}*0.5f));
    m_textRenderer.newText("clock",
        Magnum::Matrix3::projection(Magnum::Vector2{windowSize()})*
        Magnum::Matrix3::translation(Magnum::Vector2{windowSize()}*0.5f*Magnum::Vector2{1.0f,0.9f}));

    m_textUpdateTimer.reset();
    m_frameTimer.reset();
    m_wallClock.reset();
}

// -----------------------------------------------------------------------------
void CollisionSim::Application::tickEvent() {
    using namespace Magnum::Math::Literals;
    using FloatSecond = std::chrono::duration<float,std::ratio<1>>;

    float frameTimeSec{std::chrono::duration_cast<FloatSecond>(m_frameTimer.step()).count()};
    m_frameTimeSec.add(frameTimeSec);

    if (m_textUpdateTimer.stepIfElapsed(Constants::TextUpdateInterval)) {
        m_textRenderer.get("fps").renderer().render(Corrade::Utility::formatString("FPS: {:.1f}", 1.0/m_frameTimeSec.value()));
        float wallTimeSec{std::chrono::duration_cast<FloatSecond>(m_wallClock.peek()).count() * Constants::RealTimeScale};
        m_textRenderer.get("clock").renderer().render(Corrade::Utility::formatString("Time: {:.1f}s", wallTimeSec));
        m_frameTimeSec.reset();
    }

    for (Actor& actor : m_actors) {
        // Add gravity
        actor.addForce({0.0f, m_world.gravity() * actor.mass(), 0.0f});
        // Add arbitrary extra force for testing the simulation
        actor.addForce({10.0f, 0.0f, 0.0f}, {0.0f,0.0f,100.0f});
        actor.computeState(Constants::RealTimeScale * frameTimeSec);
    }
}

// -----------------------------------------------------------------------------
void CollisionSim::Application::drawEvent() {
    Magnum::GL::defaultFramebuffer.clear(
        Magnum::GL::FramebufferClear::Color |
        Magnum::GL::FramebufferClear::Depth);

    m_phongShader.setLightPositions({{1.4, 1.0, 0.75, 0.0}})
        .setAmbientColor(Magnum::Color3{0.3,0.3,0.3})
        .setProjectionMatrix(m_world.projection());
    for (Actor& actor : m_actors) {
        m_phongShader.setDiffuseColor(actor.colour())
            .setTransformationMatrix(actor.transformation())
            .setNormalMatrix(actor.transformation().normalMatrix())
            .draw(actor.mesh());
    }

    m_textRenderer.draw();

    redraw();
    swapBuffers();
}

// -----------------------------------------------------------------------------
MAGNUM_APPLICATION_MAIN(CollisionSim::Application)
