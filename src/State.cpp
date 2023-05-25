/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "State.h"
#include "Util.h"
#include <algorithm>

// -----------------------------------------------------------------------------
CollisionSim::ParallelState::ParallelState(const Magnum::Range3D& worldBounds,
                                           const std::vector<Actor>& actors,
                                           size_t numAllVertices,
                                           const sycl::queue& queue)
: numAllVertices{numAllVertices},
  worldBoundaries{queue, 6},
  actorIndices{queue, numAllVertices},
  mass{queue},
  bodyInertiaInv{queue},
  numVertices{queue},
  verticesOffset{queue},
  bodyVertices{queue, numAllVertices},
  worldVertices{queue, numAllVertices},
  translation{queue},
  rotation{queue},
  inertiaInv{queue},
  linearVelocity{queue},
  angularVelocity{queue},
  linearMomentum{queue},
  angularMomentum{queue},
  force{queue},
  torque{queue},
  wallCollisions{queue, numAllVertices},
  addLinearVelocity{queue, numAllVertices},
  addAngularVelocity{queue, numAllVertices},
  aabb{
    USMData<sycl::float2, Constants::NumActors>{queue},
    USMData<sycl::float2, Constants::NumActors>{queue},
    USMData<sycl::float2, Constants::NumActors>{queue},
  },
  sortedAABBEdges{
    USMData{queue, edgeArray(std::make_index_sequence<Constants::NumActors>{})}, // x
    USMData{queue, edgeArray(std::make_index_sequence<Constants::NumActors>{})}, // y
    USMData{queue, edgeArray(std::make_index_sequence<Constants::NumActors>{})}, // z
  },
  aabbOverlaps{queue} {

    worldBoundaries.hostContainer.assign({
        worldBounds.min()[0], worldBounds.max()[0],
        worldBounds.min()[1], worldBounds.max()[1],
        worldBounds.min()[2], worldBounds.max()[2],
    });

    size_t vertexOffset{0};
    for (size_t iActor{0}; iActor<Constants::NumActors; ++iActor) {
        const size_t numVerticesThisActor{actors[iActor].numVertices()};
        maxNumVerticesPerActor = std::max(maxNumVerticesPerActor, numVerticesThisActor);
        mass.hostContainer[iActor] = actors[iActor].mass();
        bodyInertiaInv.hostContainer[iActor] = Util::toSycl(actors[iActor].bodyInertiaInv());
        numVertices.hostContainer[iActor] = static_cast<uint16_t>(numVerticesThisActor);
        verticesOffset.hostContainer[iActor] = vertexOffset;
        translation.hostContainer[iActor] = Util::toSycl(actors[iActor].transformation_const().translation());
        rotation.hostContainer[iActor] = Util::toSycl(actors[iActor].transformation_const().rotationScaling());
        inertiaInv.hostContainer[iActor] = Util::toSycl(actors[iActor].inertiaInv());
        linearVelocity.hostContainer[iActor] = Util::toSycl(actors[iActor].linearVelocity());
        angularVelocity.hostContainer[iActor] = Util::toSycl(actors[iActor].angularVelocity());
        linearMomentum.hostContainer[iActor] = Util::toSycl(actors[iActor].linearMomentum());
        angularMomentum.hostContainer[iActor] = Util::toSycl(actors[iActor].angularMomentum());
        force.hostContainer[iActor] = Util::toSycl(actors[iActor].force());
        torque.hostContainer[iActor] = Util::toSycl(actors[iActor].torque());

        const auto& actorBodyVertices = actors[iActor].vertexPositions();
        std::fill(actorIndices.hostContainer.begin()+vertexOffset, actorIndices.hostContainer.begin()+vertexOffset+numVerticesThisActor, iActor);
        std::transform(actorBodyVertices.begin(), actorBodyVertices.end(), bodyVertices.hostContainer.begin()+vertexOffset,
                       [](const Magnum::Vector3& v) -> sycl::float3 {return Util::toSycl(v);});
        // Note: world vertices are left uninitialised as they are only calculated on the device
        vertexOffset += numVerticesThisActor;
    }
}

// -----------------------------------------------------------------------------
void CollisionSim::ParallelState::copyAllToDeviceAsync() const {
    worldBoundaries.copyToDevice();
    actorIndices.copyToDevice();
    mass.copyToDevice();
    bodyInertiaInv.copyToDevice();
    numVertices.copyToDevice();
    verticesOffset.copyToDevice();
    bodyVertices.copyToDevice();
    worldVertices.copyToDevice();
    translation.copyToDevice();
    rotation.copyToDevice();
    inertiaInv.copyToDevice();
    linearVelocity.copyToDevice();
    angularVelocity.copyToDevice();
    linearMomentum.copyToDevice();
    angularMomentum.copyToDevice();
    force.copyToDevice();
    torque.copyToDevice();
    wallCollisions.copyToDevice();
    addLinearVelocity.copyToDevice();
    addAngularVelocity.copyToDevice();
    sortedAABBEdges[0].copyToDevice();
    sortedAABBEdges[1].copyToDevice();
    sortedAABBEdges[2].copyToDevice();
}

// -----------------------------------------------------------------------------
CollisionSim::SequentialState::SequentialState(const Magnum::Range3D& worldBounds)
: worldBoundaries(worldBounds) {}
