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
                                           sycl::queue* queue)
: numAllVertices{numAllVertices},
  worldBoundaries{6, queue},
  actorIndices{numAllVertices, queue},
  mass{queue},
  bodyInertiaInv{queue},
  bodyVertices{USMData<float>{numAllVertices, queue},
               USMData<float>{numAllVertices, queue},
               USMData<float>{numAllVertices, queue}},
  translation{queue},
  rotation{queue},
  inertiaInv{queue},
  linearVelocity{queue},
  angularVelocity{queue},
  linearMomentum{queue},
  angularMomentum{queue},
  force{queue},
  torque{queue},
  wallCollisions{numAllVertices, queue},
  addLinearVelocity{numAllVertices, queue},
  addAngularVelocity{numAllVertices, queue} {

    worldBoundaries.hostContainer.assign({
        worldBounds.min()[0], worldBounds.max()[0],
        worldBounds.min()[1], worldBounds.max()[1],
        worldBounds.min()[2], worldBounds.max()[2],
    });

    size_t vertexOffset{0};
    for (size_t iActor{0}; iActor<Constants::NumActors; ++iActor) {
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
        const size_t numVerticesThisActor{actorBodyVertices[0].size()};
        std::fill(actorIndices.hostContainer.begin()+vertexOffset, actorIndices.hostContainer.begin()+vertexOffset+numVerticesThisActor, iActor);
        std::copy(actorBodyVertices[0].begin(), actorBodyVertices[0].end(), bodyVertices[0].hostContainer.begin()+vertexOffset);
        std::copy(actorBodyVertices[1].begin(), actorBodyVertices[1].end(), bodyVertices[1].hostContainer.begin()+vertexOffset);
        std::copy(actorBodyVertices[2].begin(), actorBodyVertices[2].end(), bodyVertices[2].hostContainer.begin()+vertexOffset);
        vertexOffset += numVerticesThisActor;
    }
}

// -----------------------------------------------------------------------------
void CollisionSim::ParallelState::copyAllToDeviceAsync() const {
    worldBoundaries.copyToDevice();
    actorIndices.copyToDevice();
    mass.copyToDevice();
    bodyInertiaInv.copyToDevice();
    bodyVertices[0].copyToDevice();
    bodyVertices[1].copyToDevice();
    bodyVertices[2].copyToDevice();
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

// -----------------------------------------------------------------------------
CollisionSim::SequentialState::SequentialState(const Magnum::Range3D& worldBounds)
: worldBoundaries(worldBounds) {}
