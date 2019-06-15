#ifndef ISOLATES_4_H
#define ISOLATES_4_H

#include "ArraySet.h"
#include "SparseArraySet.h"

#include <vector>
#include <map>
#include <set>
#include <utility>
#include <ctime>

////#define TIMERS
////#define SPARSE

template <typename NeighborSet> class Isolates4
{
public:
    Isolates4(std::vector<std::vector<int>> const &adjacencyArray, std::vector<double> const &weights);
    ~Isolates4();

    void RemoveAllIsolates(std::vector<int> &vIsolateVertices);

    size_t size() const { return isolates.Size(); }

    ArraySet const& GetIsolates() const { return isolates; }
    ArraySet const& GetInGraph()  const { return inGraph;  }
#ifdef SPARSE
    std::vector<SparseArraySet> const& Neighbors()  const { return neighbors;  }
#else
    std::vector<NeighborSet> const& Neighbors()  const { return neighbors;  }
#endif //SPARSE

protected: // methods
    bool RemoveIsolatedClique    (int const vertex, std::vector<int> &vIsolateVertices);

protected: // members
    std::vector<std::vector<int>> const &m_AdjacencyArray;
    std::vector<double>           const &m_vWeights;
#ifdef SPARSE
    std::vector<SparseArraySet>     neighbors;
#else
    std::vector<NeighborSet>     neighbors;
#endif // SPARSE
    ArraySet inGraph;
    ArraySet isolates;
    ArraySet remaining;
    std::vector<bool> vMarkedVertices;
#ifdef TIMERS
    clock_t timer;
    clock_t removeTimer;
    clock_t replaceTimer;
    clock_t sortDuringNextTimer;
    clock_t removeOneDuringNextTimer;
    clock_t removeDuringNextTimer;
    clock_t replaceDuringNextTimer;
#endif // TIMERS
};

#endif //ISOLATES_4_H
