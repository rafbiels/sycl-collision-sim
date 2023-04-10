/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "Application.h"
#include "Constants.h"
#include "Util.h"
#include <Corrade/Utility/FormatStl.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Corrade/Containers/StringStlView.h>
#include <Corrade/Utility/Debug.h>
#include <chrono>
#include <memory>

#include <Magnum/Math/Color.h>

// -----------------------------------------------------------------------------
CollisionSim::Application::Application(const Arguments& arguments) 
: Magnum::Platform::Application{arguments, Configuration{}.setTitle(Constants::ApplicationName)},
m_frameTimeSec{Constants::FrameTimeCounterWindow}
{
    Magnum::GL::Renderer::enable(Magnum::GL::Renderer::Feature::DepthTest);
    Magnum::GL::Renderer::enable(Magnum::GL::Renderer::Feature::FaceCulling);
    Magnum::GL::Renderer::enable(Magnum::GL::Renderer::Feature::Blending);
    Magnum::GL::Renderer::setBlendFunction(Magnum::GL::Renderer::BlendFunction::One, Magnum::GL::Renderer::BlendFunction::OneMinusSourceAlpha);
    Magnum::GL::Renderer::setBlendEquation(Magnum::GL::Renderer::BlendEquation::Add, Magnum::GL::Renderer::BlendEquation::Add);

    setMinimalLoopPeriod(0);
    setSwapInterval(0);

    m_textRenderer.newText("fps",
        Magnum::Matrix3::projection(Magnum::Vector2{windowSize()})*
        Magnum::Matrix3::translation(Magnum::Vector2{windowSize()}*0.5f));

    m_textUpdateTimer.reset();
    m_frameTimer.reset();
}

// -----------------------------------------------------------------------------
void CollisionSim::Application::tickEvent() {
    using FloatSecond = std::chrono::duration<float,std::ratio<1>>;
    m_frameTimeSec.add(std::chrono::duration_cast<FloatSecond>(m_frameTimer.step()).count());

    if (m_textUpdateTimer.stepIfElapsed(Constants::TextUpdateInterval)) {
        m_textRenderer.get("fps").renderer().render(Corrade::Utility::formatString("FPS: {:.1f}", 1.0/m_frameTimeSec.value()));
        m_frameTimeSec.reset();
    }
}

// -----------------------------------------------------------------------------
void CollisionSim::Application::drawEvent() {
    Magnum::GL::defaultFramebuffer.clear(
        Magnum::GL::FramebufferClear::Color |
        Magnum::GL::FramebufferClear::Depth);

    m_textRenderer.draw();

    redraw();
    swapBuffers();
}

// -----------------------------------------------------------------------------
MAGNUM_APPLICATION_MAIN(CollisionSim::Application)
