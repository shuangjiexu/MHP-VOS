
#include "CliquePhasedLocalSearch.h"

using namespace std;

CliquePhasedLocalSearch::CliquePhasedLocalSearch(vector<vector<int>> const &vAdjacencyArray, vector<double> const &vVertexWeights)
: PhasedLocalSearch(vAdjacencyArray,vVertexWeights)
{
    SetName("pls-clique");
    for (int vertex = 0; vertex < m_vAdjacencyArray.size(); ++vertex) {
        m_NotAdjacentToZero.Insert(vertex);
    }
}

int CliquePhasedLocalSearch::DegreeSelect(ResetableArraySet const &vertexSet) const
{
    size_t maxDegree(0);
    m_ScratchSpace.Clear();
    for (int const vertex : vertexSet) {
        if (m_vAdjacencyArray[vertex].size() > maxDegree) {
            m_ScratchSpace.Clear();
            maxDegree = m_vAdjacencyArray[vertex].size();
            m_ScratchSpace.Insert(vertex);
        } else if (m_vAdjacencyArray[vertex].size() == maxDegree) {
            m_ScratchSpace.Insert(vertex);
        }
    }

    int const vertexToReturn = *(m_ScratchSpace.begin() + rand()%m_ScratchSpace.Size());
    m_ScratchSpace.Clear();
    return vertexToReturn;
}


void CliquePhasedLocalSearch::AddToK(int const vertex)
{
#ifdef DEBUG
    cout << "Adding " << vertex << " to $K$" << endl << flush;
#endif // DEBUG
    if (m_K.Contains(vertex)) return;

    m_K.Insert(vertex);
#ifndef ALLOW_OVERLAP
    // Should they be mutually exclusive from $K$?
    m_NotAdjacentToOne.Remove(vertex);

    // definitely mutually exclusive, by definition
    m_NotAdjacentToZero.Remove(vertex);
#endif // ALLOW_OVERLAP

    // were already neighbors of $K$, now must be neighbors of vertex too
    vector<int> zeroDiffVertices;
    m_NotAdjacentToZero.IntersectInPlace(m_vAdjacencyArray[vertex], zeroDiffVertices);

    // if previously adjacent to all but one, and neighbor of newly added vertex
    // then still adjacent to all but one.
    // TODO/DS: Remove?
////    vector<int> oneDiffVertices;
////    m_NotAdjacentToOne.IntersectInPlace(m_vAdjacencyArray[vertex], oneDiffVertices);
    m_NotAdjacentToOne.IntersectInPlace(m_vAdjacencyArray[vertex]);
    // TODO/DS: check that C_0\U is empty.
    for (int const newVertex : zeroDiffVertices) {
////        if (newVertex == 18) {
////            cout << "Moving " << newVertex << " from C_0 to C_1" << endl << flush;
////        }
        m_NotAdjacentToOne.Insert(newVertex);
        m_bCheckOne = m_bCheckOne || !m_U.Contains(newVertex); // if u\in C_1 \ U
    }

////    cout << "Eject from C_1:";
////    for (int const vertex : oneDiffVertices) {
////        cout << " " << vertex;
////    }
////    cout << endl;
////
    m_dKWeight += m_vVertexWeights[vertex];

#ifdef CHECK_CONSISTENCY
    if (!IsConsistent()) {
        cout << "Line " << __LINE__ << ": Consistency check failed" << endl << flush;
    }
#endif // CHECK_CONSISTENCY
}


// starting from clique, initialize level sets and flags
void CliquePhasedLocalSearch::InitializeFromK()
{
    //Empty items that dependent on independent set, so they can be initialized.
    m_dKWeight = 0;
    m_NotAdjacentToZero.Clear();
    m_NotAdjacentToOne.Clear();

    for (int const vertex : m_K) {
        m_dKWeight += m_vVertexWeights[vertex];
    }

    m_bCheckZero = false;
    m_bCheckOne  = false;

    // check all-neighbors and all-but-one-neighbors
    for (int vertex = 0; vertex < m_vAdjacencyArray.size(); ++vertex) {
        // C_0 and C_1 don't contain vertices from K
#ifndef ALLOW_OVERLAP
        if (m_K.Contains(vertex)) continue;
#endif // ALLOW_OVERLAP
        size_t neighborCount(0);
        for (int const neighbor : m_vAdjacencyArray[vertex]) {
            if (m_K.Contains(neighbor)) neighborCount++;
        }

        if (neighborCount == m_K.Size()) {
            m_NotAdjacentToZero.Insert(vertex);
            m_bCheckZero = m_bCheckZero || !m_U.Contains(vertex);
        }

        if (neighborCount == m_K.Size()-1) { 
            m_NotAdjacentToOne.Insert(vertex);
            m_bCheckOne = m_bCheckOne || !m_U.Contains(vertex);
        }
    }

#ifdef CHECK_CONSISTENCY
    if (!IsConsistent()) {
        cout << "Line " << __LINE__ << ": Consistency check failed" << endl << flush;
    }
#endif // CHECK_CONSISTENCY
}

// same as InitializeFromK, but more efficient, iterates over $K$ instead
// of over all vertices.
// TODO/DS: currently, updateU does nothing...
void CliquePhasedLocalSearch::InitializeFromK2(bool const updateU)
{
    assert(!m_K.Empty());
    //Empty items that dependent on independent set, so they can be initialized.
    m_dKWeight = 0;
    m_ScratchSpace.Clear();
    m_NotAdjacentToZero.Clear();
    m_NotAdjacentToOne.Clear();

    m_bCheckZero = false;
    m_bCheckOne  = false;

    if (m_K.Size() == 1) {
        int const vertexInK(*m_K.begin());
        m_dKWeight = m_vVertexWeights[vertexInK];
        for (int const neighbor : m_vAdjacencyArray[vertexInK]) {
            m_NotAdjacentToZero.Insert(neighbor);
            m_bCheckZero = m_bCheckZero || !m_U.Contains(neighbor);
        }
        for (int vertex = 0; vertex < m_vAdjacencyArray.size(); ++vertex) {
            if (m_NotAdjacentToZero.Contains(vertex)) continue;
            m_NotAdjacentToOne.Insert(vertex);
            m_bCheckOne = m_bCheckOne || !m_U.Contains(vertex);
        }
        m_NotAdjacentToOne.Remove(vertexInK);

        return;
    }

    // update weights, follow neighbors, count them
    // insert into levels sets C_0 and C_1
    for (int const vertex : m_K) {
        m_dKWeight += m_vVertexWeights[vertex];

        for (int const neighbor : m_vAdjacencyArray[vertex]) {
#ifndef ALLOW_OVERLAP
            if (m_K.Contains(neighbor)) continue;
#endif // ALLOW_OVERLAP
            m_ScratchSpace.Insert(neighbor);
            m_vScratchCounters[neighbor]++; 
        }
    }

    for (int const vertex : m_ScratchSpace) {
        int const neighborCount(m_vScratchCounters[vertex]);
        if (neighborCount == m_K.Size()) {
            m_NotAdjacentToZero.Insert(vertex);
            m_bCheckZero = m_bCheckZero || !m_U.Contains(vertex);
        } else if (neighborCount == m_K.Size()-1) { 
            m_NotAdjacentToOne.Insert(vertex);
            m_bCheckOne = m_bCheckOne || !m_U.Contains(vertex);
        }

        m_vScratchCounters[vertex] = 0;
    }

    m_ScratchSpace.Clear();

#ifdef CHECK_CONSISTENCY
    if (!IsConsistent()) {
        cout << "Line " << __LINE__ << ": Consistency check failed" << endl << flush;
    }
#endif // CHECK_CONSISTENCY
}

bool CliquePhasedLocalSearch::IsConsistent() const
{
////    cout << "Checking Consistency..." << endl << flush;
    bool bConsistent(true);
    // check weight
    double weight(0.0);
    for (int const vertex : m_K) {
        weight += m_vVertexWeights[vertex];
        size_t neighborsInSet(0);
        for (int const neighbor : m_vAdjacencyArray[vertex]) {
            if (m_K.Contains(neighbor)) {
                neighborsInSet++;
            }
        }

        if (neighborsInSet != m_K.Size()-1) {
            cout << "Consistency Error!: vertex " << vertex << " has " << neighborsInSet << " neighbors in $K$, but should have " << m_K.Size()-1 << endl << flush;
        }
    }

    if (weight != m_dKWeight) {
        cout << "Consistency Error!: weight incorrect -> should be " << weight << ", is " << m_dKWeight << endl << flush;
        bConsistent = false;
    }

    // check all-neighbors and all-but-one-neighbors
    for (int vertex = 0; vertex < m_vAdjacencyArray.size(); ++vertex) {

        bool const bDebug(false);
////        bool const bDebug(vertex == 18);
        size_t neighborCount(0);
        for (int const neighbor : m_vAdjacencyArray[vertex]) {
            if (m_K.Contains(neighbor)) neighborCount++;
        }

        if (bDebug) {
            cout << vertex << ":";
            for (int const neighbor : m_vAdjacencyArray[vertex]) {
                cout << neighbor << " ";
            }
            cout << endl;

            cout << 176 << ":";
            for (int const neighbor : m_vAdjacencyArray[176]) {
                cout << neighbor << " ";
            }
            cout << endl;

            cout << " vertex " << vertex << " has " << neighborCount << " neighbors in independent set, and independent set has " << m_K.Size() << endl << flush;
        }


#ifndef ALLOW_OVERLAP
        if (m_K.Contains(vertex) && m_NotAdjacentToZero.Contains(vertex)) {
            cout << "Consistency Error!: vertex " << vertex << " is in K and C_0, but they are mutually exclusive (should only be in K?)" << endl << flush;
        }

        if (m_K.Contains(vertex) && m_NotAdjacentToOne.Contains(vertex)) {
            cout << "Consistency Error!: vertex " << vertex << " is in K and C_1, but they are mutually exclusive (should only be in K?)" << endl << flush;
        }

        bool dontRunCountCheck(m_K.Contains(vertex));
        if (dontRunCountCheck) continue;
#endif // ALLOW_OVERLAP

        if (neighborCount != m_K.Size() && m_NotAdjacentToZero.Contains(vertex)) {
            cout << "Consistency Error!: vertex " << vertex << " is in C_0, but does not belong in C_0" << endl << flush;
            bConsistent = false;
        }

        if (neighborCount == m_K.Size() && !m_NotAdjacentToZero.Contains(vertex)) {
            cout << "Consistency Error!: vertex " << vertex << " is not in C_0, but belongs in C_0" << endl << flush;
            bConsistent = false;
        }

        if (neighborCount != m_K.Size()-1 && m_NotAdjacentToOne.Contains(vertex)) {
            cout << "Consistency Error!: vertex " << vertex << " is in C_1, but does not belong in C_1" << endl << flush;
            bConsistent = false;
        }

        if (neighborCount == m_K.Size()-1 && !m_NotAdjacentToOne.Contains(vertex)) {
            cout << "Consistency Error!: vertex " << vertex << " is not in C_1, but belongs in C_1" << endl << flush;
            bConsistent = false;
        }
    }

    return bConsistent;
}

void CliquePhasedLocalSearch::ForceIntoK(int const vertex, bool const updateU)
{
////                AddToKFromOne(vertex);

////                size_t neighborsInK(0);
////                for (int const neighbor : m_vAdjacencyArray[vertex]) {
////                    if (m_K.Contains(neighbor)) {
////                        neighborsInK++;
////                    }
////                }
                // first restrict to neighborhood of $v$
    vector<int> diffSet;
    m_K.IntersectInPlace(m_vAdjacencyArray[vertex], diffSet);

////                if (neighborsInK != m_K.Size()) {
////                    cout << "Mismatch in independent set." << endl << flush;
////                }
////                if (diffSet.size() != 1) {
////                    cout << "ERROR!: diff set should be one..." << endl << flush;
////                }

    if (updateU) {
        for (int const diffVertex : diffSet) {
            m_U.Insert(diffVertex);
        }
    }

    // then add v and update helper sets.
    m_K.Insert(vertex);
    InitializeFromK2(updateU);
}

void CliquePhasedLocalSearch::AddToKFromOne(int const vertex)
{
    ForceIntoK(vertex, true /* update U*/);
}

// TODO/DS: finish and test. Not currently working, some vertices on in C_1
// that shouldn't be there.
////void PhasedLocalSearch::AddToKFromOne(int const vertex)
////{
////#ifdef DEBUG
////    cout << "Adding " << vertex << " to $K$" << endl << flush;
////#endif // DEBUG
////
////    // first restrict to neighborhood of $v$
////    vector<int> removedSet;
////    m_K.IntersectInPlace(m_vAdjacencyArray[vertex], removedSet);
////    assert (removedSet.size() == 1);
////    int const removedVertex(removedSet[0]);
////    m_NotAdjacentToOne.Remove(vertex);
////    m_K.Insert(vertex);
////    m_U.Insert(removedVertex);
////
////    // removedVertex is vertex's only neighbor not in K
////    // we remove removedVertex and add vertex.
////
////    vector<int> diffOne;
////    // if neighbor of removed vertex, and in $C_1$, then still in C_1
////    m_NotAdjacentToOne.IntersectInPlace(m_vAdjacencyArray[removedVertex], diffOne);
////
////    // nonneighbors of removedVertex are neighbors of everyone else in K
////    for (int const vertexInDiff : diffOne) {
////        m_NotAdjacentToZero.Insert(vertexInDiff);
////    }
////
////    vector<int> diffTwo;
////    // neighbors of vertex must be in C_0
////    m_NotAdjacentToZero.IntersectInPlace(m_vAdjacencyArray[vertex], diffTwo);
////
////    // non-neighbors are in C_1
////    for (int const vertexInDiff : diffTwo) {
////        m_NotAdjacentToOne.Insert(vertexInDiff);
////    }
////
////    // check neighbors of vertex, to see if they should be in C_1.
////    for (int const neighbor : m_vAdjacencyArray[vertex]) {
////#ifndef ALLOW_OVERLAP
////        if (m_K.Contains(neighbor)) continue;
////#endif //ALLOW_OVERLAP
////        // don't evaluate vertices that are already in C_1
////        if (m_NotAdjacentToOne.Contains(neighbor)) continue;
////
////        // test neighbor to see if it should be in C_1
////        size_t neighborCount(0);
////        for (int const nNeighbor : m_vAdjacencyArray[neighbor]) {
////            if (m_K.Contains(nNeighbor)) neighborCount++;
////        }
////
////        if (neighborCount == m_K.Size()-1) {
////            m_NotAdjacentToOne.Insert(neighbor);
////        }
////    }
////
////    ////TODO/DS: Update CheckZero and CheckOne
////    m_bCheckZero = !DiffIsEmpty(m_NotAdjacentToZero, m_U);
////    m_bCheckOne  = !DiffIsEmpty(m_NotAdjacentToOne,  m_U);
////
////////#ifdef CHECK_CONSISTENCY
////    if (!IsConsistent()) {
////        cout << "Line " << __LINE__ << ": Consistency check failed" << endl << flush;
////    }
////////#endif // CHECK_CONSISTENCY
////}

