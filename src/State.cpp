/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "State.h"
#include "Util.h"
#include <algorithm>

// -----------------------------------------------------------------------------
CollisionSim::State::State(const Magnum::Range3D& worldBoundaries,
                           const std::vector<Actor>& actors,
                           size_t numAllVertices,
                           sycl::queue* queue)
: numActors{actors.size()},
  numAllVertices{numAllVertices},
  worldBoundaries{6, queue},
  actorIndices{numAllVertices, queue},
  mass{actors.size(), queue},
  bodyInertiaInv{actors.size(), queue},
  bodyVertices{USMData<float>{numAllVertices, queue},
               USMData<float>{numAllVertices, queue},
               USMData<float>{numAllVertices, queue}},
  worldVertices{USMData<float>{numAllVertices, queue},
                USMData<float>{numAllVertices, queue},
                USMData<float>{numAllVertices, queue}},
  translation{actors.size(), queue},
  rotation{actors.size(), queue},
  inertiaInv{actors.size(), queue},
  linearVelocity{actors.size(), queue},
  angularVelocity{actors.size(), queue},
  linearMomentum{actors.size(), queue},
  angularMomentum{actors.size(), queue},
  force{actors.size(), queue},
  torque{actors.size(), queue},
  wallCollisions{numAllVertices, queue} {

    size_t vertexOffset{0};
    for (size_t iActor{0}; iActor<numActors; ++iActor) {
        mass.hostContainer[iActor] = actors[iActor].mass();
        bodyInertiaInv.hostContainer[iActor] = Util::toSycl(actors[iActor].bodyInertiaInv());
        translation.hostContainer[iActor] = Util::toSycl(actors[iActor].transformation_const().translation());
        inertiaInv.hostContainer[iActor] = Util::toSycl(actors[iActor].inertiaInv());
        linearVelocity.hostContainer[iActor] = Util::toSycl(actors[iActor].linearVelocity());

        const auto& actorBodyVertices = actors[iActor].vertexPositions();
        const auto& actorWorldVertices = actors[iActor].vertexPositionsWorld();
        const size_t numVerticesThisActor{actorBodyVertices[0].size()};
        std::fill(actorIndices.hostContainer.begin()+vertexOffset, actorIndices.hostContainer.begin()+vertexOffset+numVerticesThisActor, iActor);
        std::copy(actorBodyVertices[0].begin(), actorBodyVertices[0].end(), bodyVertices[0].hostContainer.begin()+vertexOffset);
        std::copy(actorBodyVertices[1].begin(), actorBodyVertices[1].end(), bodyVertices[1].hostContainer.begin()+vertexOffset);
        std::copy(actorBodyVertices[2].begin(), actorBodyVertices[2].end(), bodyVertices[2].hostContainer.begin()+vertexOffset);
        std::copy(actorWorldVertices[0].begin(), actorWorldVertices[0].end(), worldVertices[0].hostContainer.begin()+vertexOffset);
        std::copy(actorWorldVertices[1].begin(), actorWorldVertices[1].end(), worldVertices[1].hostContainer.begin()+vertexOffset);
        std::copy(actorWorldVertices[2].begin(), actorWorldVertices[2].end(), worldVertices[2].hostContainer.begin()+vertexOffset);
        vertexOffset += numVerticesThisActor;
    }
}

// -----------------------------------------------------------------------------
void CollisionSim::State::copyAllToDeviceAsync() const {
    worldBoundaries.copyToDevice();
    actorIndices.copyToDevice();
    mass.copyToDevice();
    bodyInertiaInv.copyToDevice();
    bodyVertices[0].copyToDevice();
    bodyVertices[1].copyToDevice();
    bodyVertices[2].copyToDevice();
    worldVertices[0].copyToDevice();
    worldVertices[1].copyToDevice();
    worldVertices[2].copyToDevice();
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
}
