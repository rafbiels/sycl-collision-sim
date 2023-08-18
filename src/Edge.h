/*
 * Copyright (C) 2023 Codeplay Software Limited
 * This work is licensed under the MIT License License.
 * For a copy, see https://opensource.org/licenses/MIT.
 */

#ifndef COLLISION_SIM_EDGE
#define COLLISION_SIM_EDGE

#include <Corrade/Utility/Debug.h>
#include <array>
#include <limits>
#include <string>

struct Edge {
    uint16_t actorIndex{std::numeric_limits<uint16_t>::max()};
    bool isEnd{false};
};

template <size_t... Indices>
consteval std::array<Edge, 2*sizeof...(Indices)> edgeArray(std::index_sequence<Indices...>) {
    return {{(Edge{Indices, false})..., (Edge{Indices, true})...}};
}

inline Corrade::Utility::Debug& operator<<(Corrade::Utility::Debug& s, Edge e) {
    return s << std::to_string(e.actorIndex).append(1, e.isEnd ? 'e' : 's').c_str();
}

#endif // COLLISION_SIM_EDGE
