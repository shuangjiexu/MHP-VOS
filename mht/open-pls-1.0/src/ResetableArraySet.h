#ifndef RESETABLE_ARRAY_SET_H
#define RESETABLE_ARRAY_SET_H

#include <set>
#include <vector>
#include <iostream>
#include <cassert>
#include <utility>

#include "ArraySet.h"

#define USE_RESETABLE

#ifdef USE_RESETABLE
class ResetableArraySet : public ArraySet
{
public:
    ResetableArraySet(size_t const size) : ArraySet(size)
    {
        for (size_t i = 0; i < size; ++i) {
            m_Lookup[i] = i;
            m_Elements[i] = i;
        }
    }

    ResetableArraySet() : ArraySet()
    {
    }

    virtual ~ResetableArraySet() {}

    void Resize(size_t const size)
    {
        m_Lookup.resize(size, -1);
        m_Elements.resize(size, -1);
    }

    void Reset() {
        m_iBegin = 0;
        m_iEnd = static_cast<int>(m_Elements.size())-1;
    }

    // Inserts are not allowed after saving state, as it is currently not supported.
    void Insert(int const x) {
        if (Contains(x)) return;
        assert(!m_bRemoved); // not allowed to insert and remove when saving states
        if (!m_States.empty()) m_bInserted = true;
        SwapElements(x, m_Elements[++m_iEnd]);
    }

    virtual bool Remove(int const x) {
        if (!Contains(x)) return false;
        assert(!m_bInserted); // not allowed to insert and remove when saving states
        if (!m_States.empty()) m_bRemoved = true;
        SwapElements(x, m_Elements[m_iEnd--]);
        return true;
    }

    void MoveTo(int const x, ResetableArraySet &other) {
        if (Remove(x)) other.Insert(x);
    }

    void CopyTo(int const x, ResetableArraySet &other) {
        if (Contains(x)) other.Insert(x);
    }

    void IntersectInPlace(std::vector<int> const &vOther, ResetableArraySet &other) {
        int iPutValueHere(m_iBegin);
        for (int const valueOther : vOther) {
            if (Contains(valueOther)) {
                SwapElements(valueOther, m_Elements[iPutValueHere++]);
            }
        }

        for (int index = iPutValueHere; index < m_iEnd; ++index) {
            other.Insert(m_Elements[index]);
        }
////        vRemaining.insert(vRemaining.end(), m_Elements.begin() + iPutValueHere, end());

        m_iEnd = iPutValueHere-1;
    }

    using ArraySet::IntersectInPlace;
    using ArraySet::DiffInPlace;

    void DiffInPlace(ResetableArraySet const &other) {
        // if we have the same value as the other set, then we remove it.
        for (int const valueOther : other) {
            Remove(valueOther);
        }
    }

    void DiffInPlace(std::vector<int> const &other, ResetableArraySet &intersect) {
        // if we have the same value as the other set, then we remove it.
        for (int const valueOther : other) {
            MoveTo(valueOther, intersect);
        }
    }

    bool operator==(ResetableArraySet const &that) const
    {
        if (Size() != that.Size()) return false;
        for (int const value : *this) {
            if (!that.Contains(value)) return false;
        }
        return true;
    }

    bool operator!=(ResetableArraySet const &that) const
    {
        return !(*this == that);
    }

};
#else
    typedef ArraySet ResetableArraySet;
#endif // USE_RESETABLE

#endif // RESETABLE_ARRAY_SET_H
