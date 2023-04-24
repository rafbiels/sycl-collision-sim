/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "TextRenderer.h"

#include <Magnum/FileCallback.h>
#include <Magnum/Math/Color.h>
#include <Corrade/Utility/Resource.h>
#include <Corrade/Containers/Optional.h>

// -----------------------------------------------------------------------------
CollisionSim::Text::Text(Magnum::Text::AbstractFont& font,
                         const Magnum::Text::GlyphCache& glyphCache,
                         float size,
                         Magnum::Text::Alignment alignment,
                         const Magnum::Math::Matrix3<float>& transform)
: m_renderer(std::make_unique<Magnum::Text::Renderer2D>(font, glyphCache, size, alignment)),
m_transform(transform) {}

// -----------------------------------------------------------------------------
CollisionSim::TextRenderer::TextRenderer() {
    m_importerManager.loadAndInstantiate("TgaImporter");
    m_fontManager.registerExternalManager(m_importerManager);
    m_font.reset(m_fontManager.loadAndInstantiate("MagnumFont").release());

    m_font->setFileCallback([](const std::string& filename, Magnum::InputFileCallbackPolicy, void*){
        Corrade::Utility::Resource fonts("fonts");
        return Corrade::Containers::optional(fonts.getRaw(filename));
    });
    m_font->openFile("OpenSans.conf", 0.0);

    m_glyphCache = Corrade::Containers::pointerCast<Magnum::Text::GlyphCache>(m_font->createGlyphCache());
}

// -----------------------------------------------------------------------------
bool CollisionSim::TextRenderer::newText(std::string_view key,
                                         const Magnum::Math::Matrix3<float>& transform,
                                         float size,
                                         Magnum::Text::Alignment alignment) {
    const auto [it, added] = m_texts.insert({key,
        Text{*m_font, *m_glyphCache, size, alignment, transform}});
    // TODO: move this to the caller
    it->second.renderer().reserve(50, Magnum::GL::BufferUsage::DynamicDraw, Magnum::GL::BufferUsage::StaticDraw);
    return added;
}

// -----------------------------------------------------------------------------
void CollisionSim::TextRenderer::draw() {
    using namespace Magnum::Math::Literals;
    for (auto& [key, text] : m_texts) {
        m_shader.bindVectorTexture(m_glyphCache->texture());
        m_shader.setTransformationProjectionMatrix(text.transform())
                .setColor(0xffffff_rgbf)
                .setOutlineRange(0.5f, 1.0f)
                .setSmoothness(0.075f)
                .draw(text.renderer().mesh());
    }
}

// -----------------------------------------------------------------------------
CollisionSim::Text& CollisionSim::TextRenderer::get(std::string_view key) {
    return m_texts.at(key);
}
