/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_TEXTRENDERER
#define COLLISION_SIM_TEXTRENDERER

#include <Magnum/Shaders/Shaders.h>
#include <Magnum/Text/Alignment.h>
#include <Magnum/Text/Renderer.h>
#include <Magnum/Text/AbstractFont.h>
#include <Magnum/Text/Text.h>
#include <Magnum/Text/DistanceFieldGlyphCache.h>
#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Shaders/DistanceFieldVectorGL.h>
#include <Magnum/Math/Matrix3.h>
#include <Corrade/PluginManager/Manager.h>
#include <unordered_map>
#include <string_view>
#include <memory>

namespace CollisionSim {

class Text {
    public:
        Text(Magnum::Text::AbstractFont& font,
            const Magnum::Text::GlyphCache& glyphCache,
            float size,
            Magnum::Text::Alignment alignment,
            const Magnum::Math::Matrix3<float>& transform);
        Magnum::Text::Renderer2D& renderer() {return *m_renderer;}
        Magnum::Math::Matrix3<float>& transform() {return m_transform;}
    private:
        std::unique_ptr<Magnum::Text::Renderer2D> m_renderer;
        Magnum::Math::Matrix3<float> m_transform;
};

class TextRenderer {
    public:
        TextRenderer();
        void draw();
        bool newText(std::string_view key,
                    const Magnum::Math::Matrix3<float>& transform,
                    float size=20.0,
                    Magnum::Text::Alignment alignment=Magnum::Text::Alignment::TopRight);
        Text& get(std::string_view key);
    private:
        Magnum::Shaders::DistanceFieldVectorGL2D m_shader;
        Corrade::PluginManager::Manager<Magnum::Trade::AbstractImporter> m_importerManager;
        Corrade::PluginManager::Manager<Magnum::Text::AbstractFont> m_fontManager;
        Corrade::Containers::Pointer<Magnum::Text::AbstractFont> m_font;
        Corrade::Containers::Pointer<Magnum::Text::GlyphCache> m_glyphCache;
        std::unordered_map<std::string_view, Text> m_texts;
};

} // namespace CollisionSim

#endif // COLLISION_SIM_TEXTRENDERER
