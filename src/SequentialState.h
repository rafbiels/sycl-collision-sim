/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_STATE
#define COLLISION_SIM_STATE

#include "Constants.h"
#include "Edge.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Range.h>
#include <array>
#include <unordered_set>

namespace CollisionSim {

/**
 * Class holding static data for sequential simulation
 */
class SequentialState {
    public:
        struct ActorIndexPairHash {
            size_t operator()(std::pair<uint16_t,uint16_t> p) const {
                return (static_cast<size_t>(p.first) << 16) | static_cast<size_t>(p.second);
            }
        };
        class OverlapSet : public std::unordered_set<std::pair<uint16_t,uint16_t>,ActorIndexPairHash> {};

        SequentialState() = delete;
        explicit SequentialState(const Magnum::Range3D& worldBounds)
            : worldBoundaries(worldBounds) {};

        /// Copy and assignment explicitly deleted
        ///@{
        SequentialState(const SequentialState&) = delete;
        SequentialState(SequentialState&&) = delete;
        SequentialState& operator=(const SequentialState&) = delete;
        SequentialState& operator=(SequentialState&&) = delete;
        ///}

        /// State variables
        ///@{
        Magnum::Range3D worldBoundaries;
        std::array<std::array<Edge, 2*Constants::NumActors>,3> sortedAABBEdges{
            edgeArray(std::make_index_sequence<Constants::NumActors>{}), // x
            edgeArray(std::make_index_sequence<Constants::NumActors>{}), // y
            edgeArray(std::make_index_sequence<Constants::NumActors>{}), // z
        };
        OverlapSet aabbOverlaps;
        size_t aabbOverlapsLastFrame{0};
        ///}
};

} // namespace CollisionSim

#endif // COLLISION_SIM_STATE
