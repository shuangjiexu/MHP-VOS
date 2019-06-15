
#include "PhasedLocalSearch.h"

#include <vector>

class IndependentSetPhasedLocalSearch : public PhasedLocalSearch
{
    public:
    IndependentSetPhasedLocalSearch(std::vector<std::vector<int>> const &vAdjacencyArray, std::vector<double> const &vVertexWeights);
    virtual ~IndependentSetPhasedLocalSearch() {}

    virtual int DegreeSelect(ResetableArraySet const &vertexSet) const;

    virtual void AddToK(int const vertex);

    virtual void InitializeFromK();
    virtual void InitializeFromK2(bool const updateU);

    virtual bool IsConsistent() const;

    virtual void ForceIntoK(int const vertex, bool const updateU);

    virtual void AddToKFromOne(int const vertex);
};
