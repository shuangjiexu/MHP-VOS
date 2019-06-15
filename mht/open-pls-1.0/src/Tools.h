#ifndef _DJS_TOOLS_H_
#define _DJS_TOOLS_H_

#include <list>
#include <vector>
#include <string>
#include <stdio.h>
#include <cmath>

class Algorithm;

std::vector<std::list<int>> readInGraphAdjList(int* n, int* m);

std::vector<std::list<int>> readInGraphAdjList(int &n, int &m, std::string const &fileName);
std::vector<std::list<int>> readInGraphAdjListEdgesPerLine(int &n, int &m, std::string const &fileName);

void runAndPrintStatsMatrix(long (*function)(char**,
                                             int),
                            const char* algName,
                            char** adjMatrix,
                            int n );

void RunAndPrintStats(Algorithm* pAlgorithm, std::list<std::list<int>> &cliques, bool const outputLatex);

namespace Tools
{
    std::string GetTimeInSeconds(clock_t delta, bool brackets=true);
};

#endif

