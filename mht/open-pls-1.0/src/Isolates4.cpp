#include "Isolates4.h"
#include "ArraySet.h"
#include "SparseArraySet.h"

#include <vector>
#include <set>
#include <iostream>
#include <ctime>
#include <cassert>
#include <climits>

using namespace std;

#define NEW_ISOLATE_CHECKS
////#define DOMINATION_CHECKS

template <typename NeighborSet>
Isolates4<NeighborSet>::Isolates4(vector<vector<int>> const &adjacencyArray, vector<double> const &vWeights)
 : m_AdjacencyArray(adjacencyArray)
 , m_vWeights(vWeights)
 , neighbors(adjacencyArray.size())
 , inGraph(adjacencyArray.size())
 , isolates(adjacencyArray.size())
 , remaining(adjacencyArray.size())
 , vMarkedVertices(adjacencyArray.size(), false)
#ifdef TIMERS
 , timer(0)
 , removeTimer(0)
 , replaceTimer(0)
 , sortDuringNextTimer(0)
 , removeOneDuringNextTimer(0)
 , removeDuringNextTimer(0)
 , replaceDuringNextTimer(0)
 #endif // TIMERS
{
    for (size_t u=0; u < adjacencyArray.size(); ++u) {
        remaining.Insert(u);
        inGraph.Insert(u);
#ifdef SPARSE
        neighbors[u].Resize(m_AdjacencyArray[u].size());
#else
        neighbors[u].InitializeFromAdjacencyArray(m_AdjacencyArray, u);
#endif // SPARSE
        for (int const vertex : m_AdjacencyArray[u]) {
            neighbors[u].Insert(vertex);
        }
    }
}

template <typename NeighborSet>
Isolates4<NeighborSet>::~Isolates4()
{

#ifdef TIMERS
    cout << "Total time spent computing next vertex: " << (timer/(double)CLOCKS_PER_SEC) << endl;
    cout << "    Sort      : " << (sortDuringNextTimer/(double)CLOCKS_PER_SEC) << endl;
    cout << "    RemoveOne : " << (removeOneDuringNextTimer/(double)CLOCKS_PER_SEC) << endl;
    cout << "    RemoveRest: " << (removeDuringNextTimer/(double)CLOCKS_PER_SEC) << endl;
    cout << "    ReplaceAll: " << (replaceDuringNextTimer/(double)CLOCKS_PER_SEC) << endl;
   
    cout << "Total time spent applying reductions  : " << (removeTimer/(double)CLOCKS_PER_SEC) << endl;
    cout << "Total time spent undoing  reductions  : " << (replaceTimer/(double)CLOCKS_PER_SEC) << endl;
#endif // TIMERS
}

template <typename NeighborSet>
bool Isolates4<NeighborSet>::RemoveIsolatedClique(int const vertex, vector<int> &vIsolateVertices)
{
////    bool const debug(vertex==40653);
////    if (vertex == 31) {
////        cout << "Removing isolated clique with vertex " << vertex << endl << flush;
////        if (!inGraph.Contains(31)) cout << "Vertex 31 is not in the graph!" << endl << flush;
////    }

    size_t neighborCount(0);
    for (int const neighbor : neighbors[vertex]) {
        if (neighbors[neighbor].Size() < neighbors[vertex].Size() || m_vWeights[neighbor] > m_vWeights[vertex]) {
            return false;
        }
    }

    bool superSet(true);

////    bool debug(vertex == 113);
////
////    if (debug) {
////        cout << "vertex" << vertex << " has " << neighbors[vertex].size() << " neighbors." << endl << flush;
////    }

#if 1
////    bool const debug(true); //vertex==12);
////    if (debug) {
////        cout << "vertex " << vertex << " has neighbors: ";
////    }
////    if (debug) {
////        for (int const neighbor : neighbors[vertex]) {
////            cout << neighbor << " " << flush;
////        }
////        cout << endl << flush;
////        cout << "(";
////
////        for (int const neighbor : m_AdjacencyArray[vertex]) {
////            cout << neighbor << " " << flush;
////        }
////        cout << ")" << endl << flush;
////    }
////
    for (int const neighbor : neighbors[vertex]) {
////        if (debug) {
////            cout << "Considering neighbor " <<  neighbor  << flush;
////        }
        neighborCount++;

        for (int const nNeighbor : neighbors[neighbor]) {
            vMarkedVertices[nNeighbor] = true;
        }
        vMarkedVertices[neighbor] = true;

        for (int const neighbor2 : neighbors[vertex]) {
////            if (debug && superSet && !vMarkedVertices[neighbor2]) {
////                cout << "(missing neighbor " << neighbor2 << ") ";
////            }
            superSet = superSet && vMarkedVertices[neighbor2];
        }

        //superSet = superSet && (neighborCount == neighbors[vertex].size()); // TODO/DS : put back?

        for (int const nNeighbor : neighbors[neighbor]) {
            vMarkedVertices[nNeighbor] = false;
        }
        vMarkedVertices[neighbor] = false;

        if (!superSet) {
////            if (debug) {
////                cout << "[x] " << endl;
////            }
            return false;
        }

////        if (debug) cout << endl << flush;

        ////            cout << "Iteration(" << neighborCount << ": Intersection=";
        ////            for (int const vertex : intersection) {
        ////                cout << vertex << " ";
        ////            }
        ////            cout << endl << flush;
    }
#else

    for (int const neighbor : neighbors[vertex]) {
////        cout << "Evaluating neighbor: " << neighbor << endl << flush;
        neighborCount++;

        for (int const nNeighbor : neighbors[neighbor]) {
////            cout << "Marking nNeighbor: " << nNeighbor << endl << flush;
            vMarkedVertices[nNeighbor] = true;
        }
////        cout << "Marking neighbor: " << neighbor << endl << flush;
        vMarkedVertices[neighbor] = true;

        for (int const neighbor2 : neighbors[vertex]) {
////            cout << "Checking neighbor: " << neighbor2 << endl << flush;
            superSet = superSet && vMarkedVertices[neighbor2];
        }

        for (int const nNeighbor : neighbors[neighbor]) {
////            cout << "Unmarking nNeighbor: " << nNeighbor << endl << flush;
            vMarkedVertices[nNeighbor] = false;
        }
////        cout << "Unmarking neighbor: " << neighbor << endl << flush;
        vMarkedVertices[neighbor] = false;

        if (!superSet) return false;

        ////            cout << "Iteration(" << neighborCount << ": Intersection=";
        ////            for (int const vertex : intersection) {
        ////                cout << vertex << " ";
        ////            }
        ////            cout << endl << flush;
    }
#endif

////    cout << "Done evaluating neighbors" << endl << flush;

    ////        cout << "Iteration(" << neighborCount << ": Intersection=";
    ////        for (int const vertex : intersection) {
    ////            cout << vertex << " ";
    ////        }
    ////        cout << endl << flush;

    if (superSet) {
////        if (debug) {
////            cout << "Removing Clique: [" << vertex;
////            for (int const neighbor : neighbors[vertex]) {
////                cout << " " << neighbor;
////            }
////            cout << "]" << endl;
////        }
////        for (int const neighbor : neighbors[vertex]) {
////            cout << neighbor << " ";
////        }
////        cout << endl;

////        cout << "Removing isolated vertex " << vertex << " with neighbors ";
        for (int const neighbor : neighbors[vertex]) {
////            if (!inGraph.Contains(neighbor)) {
////                cout << "Trying to remove non-existant neighbor " << neighbor << " from " << vertex << " neighbor list!" << endl << flush;
////            }
////            cout << __LINE__ << ": calling remove" << endl << flush;
            inGraph.Remove(neighbor);
            remaining.Remove(neighbor);
////            cout << neighbor << " ";
        }
////        cout << endl << flush;

////        if (!inGraph.Contains(vertex)) {
////            cout << "Trying to remove non-existant vertex " << vertex << " from graph!" << endl << flush;
////        }

////        cout << __LINE__ << ": calling remove" << endl << flush;
        inGraph.Remove(vertex);
        isolates.Insert(vertex);
        vIsolateVertices.push_back(vertex);
////        if (vertex == 0) {
////            cout << __LINE__ << ": Removing " << vertex << " from graph." << endl << flush;
////        }

        for (int const neighbor : neighbors[vertex]) {
            ////                cout << "   Expunging neighbor " << neighbor << " with " << neighbors[neighbor].size() << " neighbors" << endl << flush;
            for (int const nNeighbor : neighbors[neighbor]) {
                if (inGraph.Contains(nNeighbor)) {
                    remaining.Insert(nNeighbor);
                }

                if (nNeighbor != vertex) {
                    neighbors[nNeighbor].Remove(neighbor);
                }
            }
            neighbors[neighbor].Clear();
        }
        neighbors[vertex].Clear();

////    if (vertex == 21952) {
////        cout << "vertex" << vertex << " is being removed." << endl << flush;
////    }
        
        return true;
    }
    return false;
}

template <typename NeighborSet>
void Isolates4<NeighborSet>::RemoveAllIsolates(vector<int> &vIsolateVertices)
{
#ifdef TIMERS
    clock_t startClock = clock();
#endif // TIMERS

    if (true) {
        remaining.Clear();
        for (int const vertex : inGraph) {
            remaining.Insert(vertex);
        }
    }

    while (!remaining.Empty()) {
        int const vertex = *(remaining.begin());
        remaining.Remove(vertex);
////        if (neighbors[vertex].Empty()) {
////            inGraph.Remove(vertex);
////            isolates.Insert(vertex);
////            vIsolateVertices.push_back(vertex);
////        } else {
            RemoveIsolatedClique(vertex, vIsolateVertices);
////        }
    }

#ifdef TIMERS
    clock_t endClock = clock();
    removeTimer += (endClock - startClock);
#endif // TIMERS
}

template class Isolates4<ArraySet>;
template class Isolates4<SparseArraySet>;

