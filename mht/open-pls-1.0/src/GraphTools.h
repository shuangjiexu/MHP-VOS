#ifndef GRAPH_TOOLS_H
#define GRAPH_TOOLS_H

#include "SparseArraySet.h"
#include "ArraySet.h"

#include <vector>
#include <set>
#include <map>

namespace GraphTools
{
    void ComputeInducedSubgraph(std::vector<std::vector<int>> const &adjacencyList, std::set<int> const &vertices, std::vector<std::vector<int>> &subraph, std::map<int,int> &remapping);
    template<typename IsolatesType>
    void ComputeInducedSubgraphIsolates(IsolatesType const &isolates, std::vector<std::vector<int>> &subraph, std::map<int,int> &remapping);

    void PrintGraphInEdgesFormat(std::vector<std::vector<int>> const &adjacencyArray);
    void PrintGraphInSNAPFormat(std::vector<std::vector<int>> const &adjacencyArray);

    template<typename IsolatesType>
    void ComputeConnectedComponents(IsolatesType const &isolates, std::vector<std::vector<int>> &vComponents, size_t const uNumVertices);

    void ComputeConnectedComponents(std::vector<std::vector<int>> const &adjacencyList, std::vector<std::vector<int>> &vComponents);
};

#endif //GRAPH_TOOLS_H
