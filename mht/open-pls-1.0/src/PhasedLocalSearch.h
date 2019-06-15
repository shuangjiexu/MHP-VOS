#ifndef PLS_H 
#define PLS_H

#include "Algorithm.h"
#include "ResetableArraySet.h"

#include <string>
#include <vector>
#include <cstddef>
#include <ctime>

enum SelectionPhase {RANDOM_SELECTION, PENALTY_SELECTION, DEGREE_SELECTION };

class PhasedLocalSearch : public Algorithm
{
public:
    PhasedLocalSearch(std::vector<std::vector<int>> const &vAdjacencyArray, std::vector<double> const &vVertexWeights);
    virtual ~PhasedLocalSearch() {}

    void Perturb();

    void UpdatePenalties();

    int RandomSelect (ResetableArraySet const &vertexSet) const;
    int PenaltySelect(ResetableArraySet const &vertexSet) const;
    virtual int DegreeSelect (ResetableArraySet const &vertexSet) const = 0;

////    void RunPhase(int const iterations, SelectionPhase selectionPhase);

    int SelectFrom(ResetableArraySet const &vertexSet) const;
    int SelectFromZero();
    int SelectFromOne();

    void UpdateStatistics();

    bool Phase(size_t iterations, SelectionPhase const selectionPhase);

    virtual bool Run();
    std::string GetName() const { return m_sName; }

    void SetTargetSize(size_t const uTargetSize);

    void SetMaxSelections(size_t const uMaxSelections);

    bool DiffIsEmpty(ResetableArraySet const A, ResetableArraySet const B) const;

    virtual void AddToK(int const vertex)   = 0;
    virtual void AddToKFromOne(int const vertex) = 0;

    virtual void InitializeFromK()  = 0;
    virtual void InitializeFromK2(bool const updateU) = 0;

    virtual bool IsConsistent() const = 0;

    void SetTimeOutInMilliseconds(size_t const timeout);
    void   SetTargetWeight(double const targetWeight);
    double GetTargetWeight() { return m_dTargetWeight; }

    size_t GetSelections() const { return m_uSelections; }
    double GetBestWeight() const { return m_dBestWeight; }

    clock_t GetTimeToBestWeight() const { return m_TimeToReachBestWeight; }
    size_t  GetSelectionsToBestWeight() const { return m_uSelectionsToBestWeight; }

    void SetQuiet(bool const quiet) { m_bQuiet = quiet; }

    double GetTimeoutInSeconds() const;
    size_t GetMaxSelections() const { return m_uMaxSelections; }
    size_t GetPenaltyDelay() const;

    ResetableArraySet const& GetBestK() const;

    virtual void ForceIntoK(int const vertex, bool const updateU) = 0;

protected:
    std::string m_sName;

// Inputs.
    std::vector<std::vector<int>> const &m_vAdjacencyArray;
    std::vector<double>           const &m_vVertexWeights;

// Penalty variables.
    std::vector<int>                     m_vVertexPenalties;
    size_t                               m_uPenaltyDelay;
    size_t                               m_uNumPenalizedVertices;
    size_t                               m_uTargetPenalizedVertices;
    size_t                               m_uIterationsSinceLastPenaltyUpdate;

    size_t                               m_uTargetSize;
    double                               m_dTargetWeight;
    size_t                               m_uMaxSelections;
    size_t                               m_uSelections;

// Sets
    ResetableArraySet                    m_K;
    ResetableArraySet                    m_BestK;
    ResetableArraySet                    m_RandomK;
    ResetableArraySet                    m_DegreeK;
    ResetableArraySet                    m_U;
    ResetableArraySet                    m_NotAdjacentToOne;
    ResetableArraySet                    m_NotAdjacentToZero;
    mutable ResetableArraySet            m_ScratchSpace;
    mutable std::vector<int>             m_vScratchCounters;
    bool                                 m_bCheckZero;
    bool                                 m_bCheckOne;

// Progress Tracking
    enum SelectionPhase                  m_SelectionPhase;
    double                               m_dKWeight;
    double                               m_dBestWeight;
    size_t                               m_uBestSize;
    clock_t                              m_TimeToReachBestWeight;
    size_t                               m_uSelectionsToBestWeight;
    clock_t                              m_StartTime;
    clock_t                              m_TimeOut;

    bool                                 m_bQuiet;
};

#endif //PLS_H
