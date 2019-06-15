#ifndef CLIQUE_PHASED_LOCAL_SEARCH
#define CLIQUE_PHASED_LOCAL_SEARCH

#include "PhasedLocalSearch.h"

#include <vector>

class CliquePhasedLocalSearch : public PhasedLocalSearch
{
    public:
    CliquePhasedLocalSearch(std::vector<std::vector<int>> const &vAdjacencyArray, std::vector<double> const &vVertexWeights);
    virtual ~CliquePhasedLocalSearch() {}

    virtual int DegreeSelect(ResetableArraySet const &vertexSet) const;

    virtual void AddToK(int const vertex);

    virtual void InitializeFromK();
    virtual void InitializeFromK2(bool const updateU);

    virtual bool IsConsistent() const;

    virtual void ForceIntoK(int const vertex, bool updateU);

    virtual void AddToKFromOne(int const vertex);
};

#endif // CLIQUE_PHASED_LOCAL_SEARCH
