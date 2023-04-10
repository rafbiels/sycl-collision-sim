cmake_minimum_required(VERSION 3.3)
project(CollisionSim)

set(CMAKE_CXX_STANDARD 20)

find_package(Magnum
             REQUIRED
             GL Sdl2Application Shaders Text MagnumFont)

corrade_add_resource(CollisionSim_RESOURCES assets/resources.conf)

add_executable(collision-sim src/Application.cpp
    src/Util.cpp
    src/TextRenderer.cpp
    ${CollisionSim_RESOURCES})

target_link_libraries(collision-sim PRIVATE
    Magnum::Application
    Magnum::GL
    Magnum::Magnum
    Magnum::Shaders
    Magnum::Text
    Magnum::MagnumFont)
