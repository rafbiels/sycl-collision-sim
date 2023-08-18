cmake_minimum_required(VERSION 3.12)
project(CollisionSim)

# ------------------------------------------------
# Configure backend flags
# ------------------------------------------------
set(ENABLE_CUDA OFF CACHE BOOL "Build with CUDA target")
set(ENABLE_HIP OFF CACHE BOOL "Build with HIP target")
set(ENABLE_SPIR ON CACHE BOOL "Build with spir64 target")
set(SYCL_TARGETS "")

# ------------------------------------------------
# Configure CUDA Compute Capability
# ------------------------------------------------
if(${ENABLE_CUDA})
    string(JOIN "," SYCL_TARGETS "${SYCL_TARGETS}" "nvptx64-nvidia-cuda")
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
endif()

# ------------------------------------------------
# Configure HIP target
# ------------------------------------------------
if(${ENABLE_HIP})
    string(JOIN "," SYCL_TARGETS "${SYCL_TARGETS}" "amdgcn-amd-amdhsa")
    set(DEFAULT_HIP_GFX_ARCH "gfx906")
    execute_process(
        COMMAND bash -c "which rocminfo >/dev/null && rocminfo | grep -o 'gfx[0-9]*' | head -n 1"
        OUTPUT_VARIABLE HIP_GFX_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if ("${HIP_GFX_ARCH}" STREQUAL "")
        message(WARNING "Failed to autoconfigure CUDA Compute Capability using nvidia-smi. Will default to sm_${DEFAULT_HIP_GFX_ARCH}")
        set(HIP_GFX_ARCH ${DEFAULT_HIP_GFX_ARCH})
    else()
        message(STATUS "Detected HIP gfx arch to ${HIP_GFX_ARCH}")
    endif()
endif()

# ------------------------------------------------
# Configure spir64 target
# ------------------------------------------------
if(${ENABLE_SPIR})
    string(JOIN "," SYCL_TARGETS "${SYCL_TARGETS}" "spir64")
endif()

# ------------------------------------------------
# Configure headless or display mode
# ------------------------------------------------
set(HEADLESS OFF CACHE BOOL "Build without window display")
if(${HEADLESS})
    message(STATUS "Configuring the application without display because HEADLESS=ON")
    set(MagnumApplicationModule WindowlessGlxApplication)
    set(MagnumApplicationLibrary Magnum::WindowlessGlxApplication)
    add_compile_definitions(HEADLESS=true)
else()
    message(STATUS "Configuring the application with SDL2 window display because HEADLESS=OFF")
    set(MagnumApplicationModule Sdl2Application)
    set(MagnumApplicationLibrary Magnum::Application)
    add_compile_definitions(HEADLESS=false)
endif()

# ------------------------------------------------
# Configure the size of the actor grid
# ------------------------------------------------
set(ACTOR_GRID_SIZE "5" CACHE STRING "Size of the square grid of actors")
add_compile_definitions(ACTOR_GRID_SIZE=${ACTOR_GRID_SIZE})

# ------------------------------------------------
# Configure compilation flags
# ------------------------------------------------
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O1")
set(SYCL_FLAGS -fsycl -fsycl-targets=${SYCL_TARGETS})
if(${ENABLE_CUDA})
    set(SYCL_FLAGS ${SYCL_FLAGS} -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_${CUDA_COMPUTE_CAPABILITY})
endif()
if(${ENABLE_HIP})
    set(SYCL_FLAGS ${SYCL_FLAGS} -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${HIP_GFX_ARCH})
endif()

# ------------------------------------------------
# Configure Magnum/Corrade
# ------------------------------------------------
find_package(Magnum
             REQUIRED
             GL ${MagnumApplicationModule} Primitives Shaders Text MagnumFont)
corrade_add_resource(CollisionSim_RESOURCES assets/resources.conf)

# ------------------------------------------------
# Configure the sources and linking for individual objects
# ------------------------------------------------
# Targets with SYCL dependency
add_library(Util OBJECT src/Util.cpp)
target_compile_options(Util PUBLIC ${SYCL_FLAGS})
target_link_libraries(Util
    PUBLIC Magnum::Magnum)

add_library(ParallelState OBJECT src/ParallelState.cpp)
target_compile_options(ParallelState PUBLIC ${SYCL_FLAGS})
target_link_libraries(ParallelState
    PUBLIC Magnum::Magnum
    PRIVATE Util Actor)

add_library(ParallelSimulation OBJECT src/ParallelSimulation.cpp)
target_compile_options(ParallelSimulation PUBLIC ${SYCL_FLAGS})
target_link_libraries(ParallelSimulation
    PRIVATE Util Actor Magnum::Magnum
    PUBLIC ParallelState)

# Targets without SYCL dependency
add_library(SequentialSimulation OBJECT src/SequentialSimulation.cpp)
target_link_libraries(SequentialSimulation
    PRIVATE Util Actor
    PUBLIC Magnum::Magnum)

add_library(Shape OBJECT src/Shape.cpp)
target_link_libraries(Shape
    PUBLIC Magnum::Magnum Magnum::GL)

add_library(Actor OBJECT src/Actor.cpp)
target_link_libraries(Actor
    PRIVATE Util
    PUBLIC Shape Magnum::Magnum Magnum::Primitives)

add_library(TextRenderer OBJECT src/TextRenderer.cpp)
target_link_libraries(TextRenderer
    PUBLIC Magnum::Magnum Magnum::Text Magnum::MagnumFont)

add_library(World OBJECT src/World.cpp)
target_link_libraries(World
    PUBLIC Shape Magnum::Magnum Magnum::Primitives)

# ------------------------------------------------
# Configure the sources and linking for the executable
# ------------------------------------------------
add_executable(collision-sim src/Application.cpp
    ${CollisionSim_RESOURCES})

target_link_libraries(collision-sim
    PRIVATE
    Actor ParallelSimulation ParallelState SequentialSimulation Shape TextRenderer Util World
    ${MagnumApplicationLibrary}
    Magnum::GL Magnum::Magnum Magnum::Shaders)

target_compile_options(collision-sim PUBLIC ${SYCL_FLAGS})
target_link_options(collision-sim PUBLIC ${SYCL_FLAGS})
