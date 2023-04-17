cmake_minimum_required(VERSION 3.12)
project(CollisionSim)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80")

find_package(Magnum
             REQUIRED
             GL Sdl2Application Primitives Shaders Text MagnumFont)

corrade_add_resource(CollisionSim_RESOURCES assets/resources.conf)

add_executable(collision-sim src/Application.cpp
    src/Actor.cpp
    src/CollisionCalculator.cpp
    src/Shape.cpp
    src/TextRenderer.cpp
    src/Util.cpp
    src/World.cpp
    ${CollisionSim_RESOURCES})

target_link_libraries(collision-sim PRIVATE
    Magnum::Application
    Magnum::GL
    Magnum::Magnum
    Magnum::Primitives
    Magnum::Shaders
    Magnum::Text
    Magnum::MagnumFont)
