cmake_minimum_required(VERSION 3.12)
project(CollisionSim)

set(DEFAULT_CUDA_COMPUTE_CAPABILITY "50")
execute_process(
    COMMAND bash -c "which nvidia-smi >/dev/null && nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.'"
    OUTPUT_VARIABLE CUDA_COMPUTE_CAPABILITY
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if ("${CUDA_COMPUTE_CAPABILITY}" STREQUAL "")
    message(WARNING "Failed to autoconfigure CUDA Compute Capability using nvidia-smi. Will default to sm_${DEFAULT_CUDA_COMPUTE_CAPABILITY}")
    set(CUDA_COMPUTE_CAPABILITY ${DEFAULT_CUDA_COMPUTE_CAPABILITY})
else()
    message(STATUS "Detected CUDA Compute Capability to sm_${CUDA_COMPUTE_CAPABILITY}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_${CUDA_COMPUTE_CAPABILITY}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O1 -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-noopt-device-debug")

find_package(Magnum
             REQUIRED
             GL Sdl2Application Primitives Shaders Text MagnumFont)

corrade_add_resource(CollisionSim_RESOURCES assets/resources.conf)

add_executable(collision-sim src/Application.cpp
    src/Actor.cpp
    src/Shape.cpp
    src/Simulation.cpp
    src/State.cpp
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
