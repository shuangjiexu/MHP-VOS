#include "GraphTools.h"
#include "SparseArraySet.h"
#include "ArraySet.h"
#include "Isolates4.h"

#include <set>
#include <vector>
#include <list>
#include <map>
#include <iostream>
#include <algorithm>

using namespace std;

void GraphTools::ComputeInducedSubgraph(vector<vector<int>> const &graph, set<int> const &vertices, vector<vector<int>> &subgraph, map<int,int> &remapping)
{
    subgraph.clear();
    remapping.clear();

////    cout << "Forming induced subgraph on " << vertices.size() << " vertices." << endl;

    map<int,int> forwardMapping;

    int vertexIndex(0);
    auto mappedVertex = [&vertexIndex, &remapping, &forwardMapping](int const vertex)
    {
        if (forwardMapping.find(vertex) == forwardMapping.end()) {
            forwardMapping[vertex] = vertexIndex;
            remapping[vertexIndex] = vertex;
            vertexIndex++;
        }
        return forwardMapping[vertex];
    };

    for (int const vertex : vertices) {
        mappedVertex(vertex);
    }

    subgraph.resize(vertices.size());

    for (int vertex = 0; vertex < graph.size(); ++vertex) {
        if (vertices.find(vertex) == vertices.end()) continue;

        vector<int> const &neighbors(graph[vertex]);
        int const newVertex = mappedVertex(vertex);
////        cout << newVertex << " : ";
        for (int const neighbor : neighbors) {
            if (vertices.find(neighbor) == vertices.end()) continue;
            int const newNeighbor = mappedVertex(neighbor);
            subgraph[newVertex].push_back(newNeighbor);
////            subgraph[newNeighbor].push_back(newVertex);
////            cout << newNeighbor << " ";
        }
////        cout << endl;
    }
}

template <typename IsolatesType>
void GraphTools::ComputeInducedSubgraphIsolates(IsolatesType const &isolates, vector<vector<int>> &subgraph, map<int,int> &remapping)
{
    subgraph.clear();
    remapping.clear();

    map<int,int> forwardMapping;

    int vertexIndex(0);
    auto mappedVertex = [&vertexIndex, &remapping, &forwardMapping](int const vertex)
    {
        if (forwardMapping.find(vertex) == forwardMapping.end()) {
            forwardMapping[vertex] = vertexIndex;
            remapping[vertexIndex] = vertex;
            vertexIndex++;
        }
        return forwardMapping[vertex];
    };

    for (int const vertex : isolates.GetInGraph()) {
        mappedVertex(vertex);
    }

    subgraph.resize(isolates.GetInGraph().Size());

    for (int const vertex : isolates.GetInGraph()) {
        if (!isolates.GetInGraph().Contains(vertex)) continue;

        int const newVertex = mappedVertex(vertex);
////        cout << newVertex << " : ";
        for (int const neighbor : isolates.Neighbors()[vertex]) {
            if (!isolates.GetInGraph().Contains(neighbor)) continue;
            int const newNeighbor = mappedVertex(neighbor);
            subgraph[newVertex].push_back(newNeighbor);
////            subgraph[newNeighbor].push_back(newVertex);
////            cout << newNeighbor << " ";
        }
////        cout << endl;
    }
}

template<typename IsolatesType>
void GraphTools::ComputeConnectedComponents(IsolatesType const &isolates, vector<vector<int>> &vComponents, size_t const uNumVertices) {
    ArraySet remaining = isolates.GetInGraph();

    ArraySet currentSearch(uNumVertices);
    vector<bool> evaluated(uNumVertices, 0);

    size_t componentCount(0);
    vComponents.clear();

    if (!remaining.Empty()) {
        int const startVertex = *remaining.begin();
        currentSearch.Insert(startVertex);
        remaining.Remove(startVertex);
        componentCount++;
        vComponents.resize(componentCount);
    }

    while (!remaining.Empty() && !currentSearch.Empty()) {
        int const nextVertex(*currentSearch.begin());
        evaluated[nextVertex] = true;
        vComponents[componentCount - 1].push_back(nextVertex);
        currentSearch.Remove(nextVertex);
        remaining.Remove(nextVertex);
        for (int const neighbor : isolates.Neighbors()[nextVertex]) {
            if (!evaluated[neighbor]) {
                currentSearch.Insert(neighbor);
            }
        }

        if (currentSearch.Empty() && !remaining.Empty()) {
            int const startVertex = *remaining.begin();
            currentSearch.Insert(startVertex);
            remaining.Remove(startVertex);
            componentCount++;
            vComponents.resize(componentCount);
        }
    }
}

void GraphTools::ComputeConnectedComponents(vector<vector<int>> const &adjacencyList, vector<vector<int>> &vComponents) {

    vComponents.clear();
    if (adjacencyList.empty()) return;


    size_t componentCount(0);
    size_t uNumVertices(adjacencyList.size());

    vector<bool> evaluated    (uNumVertices, false);
    ArraySet     currentSearch(uNumVertices);
    ArraySet     remaining    (uNumVertices);

    for (int vertex = 0; vertex < uNumVertices; ++vertex) {
        remaining.Insert(vertex);
    }

    // add first vertex, from where we start search
    int const startVertex(0);
    currentSearch.Insert(startVertex);
    remaining.Remove(startVertex);
    componentCount++;
    vComponents.resize(componentCount);

    while (!remaining.Empty() && !currentSearch.Empty()) {
        int const nextVertex(*currentSearch.begin());
        evaluated[nextVertex] = true;
        vComponents[componentCount - 1].push_back(nextVertex);
        currentSearch.Remove(nextVertex);
        remaining.Remove(nextVertex);
        for (int const neighbor : adjacencyList[nextVertex]) {
            if (!evaluated[neighbor]) {
                currentSearch.Insert(neighbor);
            }
        }

        if (currentSearch.Empty() && !remaining.Empty()) {
            int const startVertex = *remaining.begin();
            currentSearch.Insert(startVertex);
            remaining.Remove(startVertex);
            componentCount++;
            vComponents.resize(componentCount);
        }
    }
}

void GraphTools::PrintGraphInEdgesFormat(vector<vector<int>> const &adjacencyArray)
{
    cout << adjacencyArray.size() << endl;
    size_t edges(0);
    for (vector<int> const &neighborList : adjacencyArray) {
        edges+= neighborList.size();
    }
    cout << edges << endl;

    for (size_t index = 0; index < adjacencyArray.size(); ++index) {
        for (int const neighbor : adjacencyArray[index]) {
            cout << index << "," << neighbor << endl;
        }
    }
}

void GraphTools::PrintGraphInSNAPFormat(vector<vector<int>> const &adjacencyArray)
{
    for (size_t index = 0; index < adjacencyArray.size(); ++index) {
        for (int const neighbor : adjacencyArray[index]) {
            cout << (index+1) << " " << (neighbor+1) << endl << flush;
        }
    }
}

template
void GraphTools::ComputeInducedSubgraphIsolates<Isolates4<SparseArraySet>>(Isolates4<SparseArraySet> const &isolates, vector<vector<int>> &subgraph, map<int,int> &remapping);

template
void GraphTools::ComputeConnectedComponents<Isolates4<SparseArraySet>>(Isolates4<SparseArraySet> const &isolates, vector<vector<int>> &vComponents, size_t const uNumVertices);

template
void GraphTools::ComputeConnectedComponents<Isolates4<ArraySet>>(Isolates4<ArraySet> const &isolates, vector<vector<int>> &vComponents, size_t const uNumVertices);

