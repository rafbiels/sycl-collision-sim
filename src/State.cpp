/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#include "State.h"
#include "Util.h"
#include <algorithm>

// -----------------------------------------------------------------------------
CollisionSim::State::State(const Magnum::Range3D& worldBounds,
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
  wallCollisions{numAllVertices, queue},
  addLinearVelocity{numAllVertices, queue},
  addAngularVelocity{numAllVertices, queue} {

    worldBoundaries.hostContainer.assign({
        worldBounds.min()[0], worldBounds.max()[0],
        worldBounds.min()[1], worldBounds.max()[1],
        worldBounds.min()[2], worldBounds.max()[2],
    });

    size_t vertexOffset{0};
    for (size_t iActor{0}; iActor<numActors; ++iActor) {
        mass.hostContainer[iActor] = actors[iActor].mass();
        bodyInertiaInv.hostContainer[iActor] = Util::toSycl(actors[iActor].bodyInertiaInv());
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
    addLinearVelocity.copyToDevice();
    addAngularVelocity.copyToDevice();
}
