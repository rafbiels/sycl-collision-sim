# SYCL Collision Simulation
Demo 3D simulation of rigid body physics with different shapes bouncing off each
other confined in a box. Two implementations are provided, one sequential with
standard C++ code compiled for CPU, and parallel
[SYCL](https://www.khronos.org/sycl/) implementation which can be compiled for
any target device (e.g. a GPU) supported by a SYCL compiler.

The main aim of this project is to showcase how SYCL can be used to port CPU
code to GPUs with standard C++ syntax, achieving significant performance boost
portable across devices from different vendors.

There is a video display with FPS counter for demonstration purposes (presented
in the clip below), as well as a headless mode for benchmarking.

<a href="https://player.vimeo.com/video/873721038">
<img src="https://i.vimeocdn.com/video/1737354781-487f33094af9fd551060d0b2df485ea656d513fdfe10a8e933ef2fc047aa14cd-d" width="50%" />
</a>


## How to build
### Requirements

* [CMake](https://cmake.org/) 3.12 or newer.
* [Intel oneAPI DPC++](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)
compiler or its
[open source version](https://github.com/intel/llvm). Although most, if not
all, of the code should compile with other SYCL implementations, the CMake
configuration assumes the DPC++ compiler driver CLI for compilation flags setup.
* [Magnum](https://doc.magnum.graphics/magnum/getting-started.html#getting-started-setup-install)
(and its dependency
[Corrade](https://doc.magnum.graphics/corrade/building-corrade.html#building-corrade-packages)) -
graphics middleware library used for the display as well as defining and
transforming actor body meshes.
* [SDL2](https://wiki.libsdl.org/SDL2/Installation) for the graphical output
(not needed when compiling for the headless mode).

### Build with CMake

Like any standard CMake project, the collision simulation can be built with
simply cloning the source, entering its directory, and typing:
```sh
mkdir build && cd build
CXX=clang++ cmake ..
cmake --build .
```
The `CXX` variable should point to your DPC++ compiler, either the `clang++`
driver or the `icpx` (they have matching CLI, so both work).

There are several options you may want to set when configuring the project. Use
them by adding `-D<option>=<value>` in the first `cmake` command, with `<value>`
being either `ON` or `OFF` for boolean options, and a number/string for others.
The available options are:
* `HEADLESS` - boolean option switching to the headless mode without graphical
output, useful for benchmarking. The default is `OFF` (meaning graphical output
is enabled).
* `ACTOR_GRID_SIZE` - integer option. The simulation runs for a square grid of
actors NxN where N=`ACTOR_GRID_SIZE`. The default value is 5 (meaning 25 actors
are simulated). Due to some compile-time calculations based on this number, the
compiler may prevent you from setting it too high. The DPC++ version 2023.2.1
allows 19 as the maximum value.
* `ENABLE_CUDA` - boolean option enabling the compilation of SYCL code for
NVIDIA GPU targets. The default value is `OFF`. Enabling this option requires
the CUDA toolkit to be installed (at least the ptx assembler and device bitcode library).
* `ENABLE_HIP` - boolean option enabling the compilation of SYCL code for
AMD GPU targets. The default value is `OFF`. Enabling this option requires
the ROCm toolkit to be installed (at least the device bitcode library).
* `ENABLE_SPIR` - boolean option enabling the compilation of SYCL code for
Intel devices. The compiled code can run both on Intel OpenCL and Level-Zero
backends. The default value is `ON`.

## How to run

The application can be simply executed with `./collision-sim`. This executes the
parallel simulation using SYCL. There is a single command-line option supported,
`--cpu`, which causes the program to execute the sequential CPU implementation
instead.

The SYCL implementation uses the default device selector, thus, the DPC++ SYCL
runtime
[environment variables](https://intel.github.io/llvm-docs/EnvironmentVariables.html)
may be used to influence its behaviour. In particular, the
`ONEAPI_DEVICE_SELECTOR` variable allows to pick a specific device among all
available ones.


## Algorithms

The physics simulation is implemented in `src/SequentialSimulation.cpp` for the
sequential C++ code and in `src/ParallelSimulation.cpp` for the SYCL version.
In addition `src/Util.cpp` implements some of the common computation. All other
files in the project only define the data structures and deal with the graphical
display and application flow. The two simulation files implement the same
general algorithms, albeit with some adjustments relevant to either CPU or GPU
programming.


#### Rigid body motion
The implementation of Newton-Euler equations describing 3D rigid body motion
follows broadly  
* D. Baraff, 2001, [*Rigid Body Simulation*](https://graphics.pixar.com/pbm2001/pdf/notesg.pdf)

The sequential simulation implements the algorithm in the
`simulateMotionSequential` function, whereas the parallel simulation covers it
in the `ActorKernel` which updates the full-body kinematic properties and
`VertexKernel` which updates all body mesh vertex positions accordingly.

#### World boundary collisions
The world boundary collision detection runs for every vertex of each actor body
mesh and compares its location to the world edges. The first vertex detected to
intersect with a world edge triggers a collision response implemented using the
impulse-based model described in:
* Wikipedia article *Collision response*, section
[*Impulse-based contact model*](https://en.wikipedia.org/wiki/Collision_response#Impulse-based_contact_model)
[accessed 10/2023]

where one of the colliding bodies is a wall with an infinite mass.

The sequential simulation (function `collideWorldSequential`) loops over all
actors, and for each actor loops over all vertices. An early-exit condition is
triggered in the inner loop on the first detected collision. The parallel
simulation implements the world collision detection as part of the
`VertexKernel` and stores the collision information for every vertex in memory.
This is then copied to host at the end of a simulation step, where the reduction
to per-actor information and the impulse application are executed on the CPU.

#### Broad-phase collision detection
The broad-range collision detection is applied to actors' axis-aligned bounding
boxes (AABB) following the basic sweep and prune algorithm described in chapter
2 of:
* D.J. Tracy, S.R. Buss, B.M. Woods, 2009,
[Efficient Large-Scale Sweep and Prune Methods with AABB Insertion and Removal](https://mathweb.ucsd.edu/~sbuss/ResearchWeb/EnhancedSweepPrune/SAP_paper_online.pdf)

which consists of three stages:
* calculation of the axis-aligned bounding box (AABB) for each actor
* sorting AABB edges along each axis
* overlap detection among the sorted edges and flagging those overlapping along all three axes

For the sequential implementation, the first stage is part of the actor's
`updateVertexPositions` procedure called in `simulateMotionSequential` and the
other two stages are implemented in `collideBroadSequential`. The parallel
version implements the three stages in the kernels: `AABBKernel`,
`AABBSortKernel`, `AABBOverlapKernel`, respectively.

The sequential implementation follows the paper closely, sorting AABB edges
along each axis using insertion sort. This approach is highly optimal for CPU,
but challenging to implement reasonably for SIMT architectures like GPUs. For
this reason, the parallel simulation uses the odd-even merge-sort algorithm in
the sorting step of the algorithm.

#### Narrow-phase collision detection
All actor pairs flagged as overlapping by the broad-phase algorithm are subject
to narrow-phase collision detection. For each such pair of actors (A, B), the
algorithm searches for the closest triangle-vertex pair between all triangles of
actor A and all vertices of actor B, as well as between all triangles of actor B
and all vertices of actor A.

The 3D triangle-vertex distance is calculated by transforming the problem into
a 2D space following the "2D method" described in:
* M.W. Jones, 1995,
[*3D Distance from a Point to a Triangle*](http://www-compsci.swan.ac.uk/~csmark/PDFS/1995_3D_distance_point_to_triangle)

The "2D method" functions are implemented in `src/Util.cpp` and the same code is
used in the sequential and parallel implementation, exploiting the flexibility
of SYCL to compile into both device and host code. The `triangleTransform`
function transforms a triangle-vertex pair into the coordinate system described
in the paper, whereas the `closestPointOnTriangle` function finds a point within
the triangle boundaries that is the closest to the vertex, and returns the
squared distance between them. Finally, the `closestPointOnTriangleND` function
applies this to an array of vertices for one triangle, and finds the vertex
closest to that triangle.

The pair of actors is flagged as colliding whenever the closest triangle-vertex
distance between them is below a fixed threshold encoded in
`Constants::NarrowPhaseCollisionThreshold`.

The triangle and vertex loop logic and threshold application is implemented in
the `collideNarrowSequential` function for the sequential simulation. The
parallel version starts with the `NarrowPhaseKernel` where each thread processes
a single triangle, pairing it with all vertices of the other actor. This is
followed by the `TVReduceKernel` which reduces all the triangle-vertex pairs for
each actor into a single one with the closest distance squared.

The parallel algorithm kernels' iteration space is calculated dynamically from
the broad-phase results and the computation is launched only for the overlapping
actor pairs. Due to this, there is a synchronisation point where the broad-phase
results are copied to host and processed to define the narrow-phase iteration
range.

#### Collision response
The collision response uses the impulse-based model described in:
* Wikipedia article *Collision response*, section
[*Impulse-based contact model*](https://en.wikipedia.org/wiki/Collision_response#Impulse-based_contact_model)
[accessed 10/2023]

The sequential simulation implements it in the `impulseCollision` function and
the parallel version in the `ImpulseCollisionKernel`.

## Known issues

#### Sticky collisions
In some cases, when clipping occurs between the colliding shapes, the impulse
collision algorithm may result in the two objects "sticking" to each other for
some amount of time. This is a common flaw of the impulse-based collision
response model, however, no workaround has been implemented so far in this
project.

#### Segfault in HEADLESS mode without display (X11)
Even though the headless mode doesn't produce any graphical output, a bug in
the underlying X11 library initialisation causes the application to crash on
startup when there is no display configured. This can be worked around on X11
Linux with the X virtual framebuffer (`Xvfb`):
```sh
Xvfb :1 -screen 0 1024x768x24 -fbdir $(mktemp -d) &
export DISPLAY=:1
./collision-sim
```
