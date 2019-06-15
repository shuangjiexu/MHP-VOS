// local includes
#include "CliquePhasedLocalSearch.h"
#include "IndependentSetPhasedLocalSearch.h"
////#include "PhasedLocalSearch.h"
#include "Algorithm.h"
#include "Tools.h"

#include "GraphTools.h"
#include "Isolates4.h"
#include "SparseArraySet.h"

// system includes
#include <map>
#include <list>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

#include <cassert>
#include <cstdlib>
#include <ctime>
////#include <climits>
#include <limits>

#define str(x) xstr(x)
#define xstr(x) #x

#define GIT_COMMIT_STRING str(GIT_COMMIT)
#define GIT_STATUS_STRING str(GIT_STATUS)
//#define EXPERIMENTAL
#define VERSION "1.0"

using namespace std;

void ProcessCommandLineArgs(int const argc, char** argv, map<string,string> &mapCommandLineArgs)
{
    for (int i = 1; i < argc; ++i) {
////        cout << "Processing argument " << i << endl;
        string const argument(argv[i]);
////        cout << "    Argument is " << argument << endl;
        size_t const positionOfEquals(argument.find_first_of("="));
////        cout << "    Position of = " << positionOfEquals << endl;
        if (positionOfEquals != string::npos) {
            string const key  (argument.substr(0,positionOfEquals));
            string const value(argument.substr(positionOfEquals+1));
////            cout << "    Parsed1: " << key << "=" << value << endl;
            mapCommandLineArgs[key] = value;
        } else {
////            cout << "    Parsed2: " << argument << endl;
            mapCommandLineArgs[argument] = "";
        }
    }
}

void PrintDebugWarning()
{
    cout << "\n\n\n\n\n" << flush;
    cout << "#########################################################################" << endl << flush;
    cout << "#                                                                       #" << endl << flush;
    cout << "#    WARNING: Debugging is turned on. Don't believe the run times...    #" << endl << flush;
    cout << "#                                                                       #" << endl << flush;
    cout << "#########################################################################" << endl << flush;
    cout << "\n\n\n\n\n" << flush;
}

void PrintExperimentalWarning()
{

#ifdef EXPERIMENTAL
    cout << "#WARNING: Phased Local Search v" << VERSION << "beta. (Experimental)" << endl << flush;
    cout << "#WARNING: " << endl;
    cout << "#WARNING: Proceed with caution: this software is currently in an experimental state." << endl << flush;
    cout << "#WARNING: This software may be slow, the algorithm may be unstable and the results may be incorrect." << endl << flush;
    cout << "#WARNING: If you care about this sort of thing, don't use it." << endl << flush;
#else
    cout << "#Phased Local Search v" << VERSION << "." << endl << flush;
#endif // EXPERIMENTAL
}

void RunUnitTests()
{
    std::cout << "#Running unit tests...";
    std::cout << "#Done!" << std::endl;
}


string basename(string const &fileName)
{
    string sBaseName(fileName);

    size_t const lastSlash(sBaseName.find_last_of("/\\"));
    if (lastSlash != string::npos) {
        sBaseName = sBaseName.substr(lastSlash+1);
    }

    size_t const lastDot(sBaseName.find_last_of("."));
    if (lastDot != string::npos) {
        sBaseName = sBaseName.substr(0, lastDot);
    }

    return sBaseName;
}

bool ReadSolution(string const &solutionFile, vector<bool> &vSolution)
{
    ifstream instream(solutionFile.c_str());

    int vertex(-1);
    double weight(-1.0); // endvertices, to read edges.
    size_t uNumRead(0);
    while (instream.good() && !instream.eof()) {
        string line;
        std::getline(instream, line);
        stringstream strm(line);
////        bool debug(true); ////u == 40656 || u == 40653);
////if (debug)        cout << (u+1) << " : " << endl << flush;
////        if (debug)        cout << "Read     Line: " << line << endl << flush;
////if (debug)        cout << "Actually Read: ";
        if (!line.empty() && strm.good() && !strm.eof()) {
            strm >> vertex;
            assert(vertex > -1 && vertex < vSolution.size());
            vSolution[vertex] = true;
////            if (debug)        cout << "Inserted     : " << vertex << "->" << weight<< endl << flush;
            uNumRead++;
////if (debug)        cout << "Actually Read: ";
        }
    }

#ifdef DEBUG
    printArrayOfLinkedLists(adjList, n);
#endif

    return true;
}


bool ReadWeights(string const &inputFileName, vector<double> &vVertexWeights)
{
    string const weightFileName(inputFileName + ".weights");
    ifstream instream(weightFileName.c_str());

    int vertex(-1);
    double weight(-1.0); // endvertices, to read edges.
    size_t uNumRead(0);
    while (instream.good() && !instream.eof()) {
        string line;
        std::getline(instream, line);
        stringstream strm(line);
////        bool debug(true); ////u == 40656 || u == 40653);
////if (debug)        cout << (u+1) << " : " << endl << flush;
////        if (debug)        cout << "Read     Line: " << line << endl << flush;
////if (debug)        cout << "Actually Read: ";
        if (!line.empty() && strm.good() && !strm.eof()) {
            strm >> vertex >> weight;
            assert(vertex > -1 && vertex < vVertexWeights.size());
            vVertexWeights[vertex] = weight;
////            if (debug)        cout << "Inserted     : " << vertex << "->" << weight<< endl << flush;
            uNumRead++;
////if (debug)        cout << "Actually Read: ";
        }
    }

#ifdef DEBUG
    printArrayOfLinkedLists(adjList, n);
#endif

    return (uNumRead == vVertexWeights.size());
}

int main(int argc, char** argv)
{
    int failureCode(0);

    map<string,string> mapCommandLineArgs;

    ProcessCommandLineArgs(argc, argv, mapCommandLineArgs);

    bool   const bQuiet(mapCommandLineArgs.find("--verbose") == mapCommandLineArgs.end());
    bool   const bOutputLatex(mapCommandLineArgs.find("--latex") != mapCommandLineArgs.end());
    bool   const bOutputTable(mapCommandLineArgs.find("--table") != mapCommandLineArgs.end());
    bool   const bWeighted(mapCommandLineArgs.find("--weighted") != mapCommandLineArgs.end());
    bool   const bUseWeightFile(mapCommandLineArgs.find("--use-weight-file") != mapCommandLineArgs.end());
    string const inputFile((mapCommandLineArgs.find("--input-file") != mapCommandLineArgs.end()) ? mapCommandLineArgs["--input-file"] : "");
    string const solutionFile((mapCommandLineArgs.find("--solution-file") != mapCommandLineArgs.end()) ? mapCommandLineArgs["--solution-file"] : "");
    string const sAlgorithm((mapCommandLineArgs.find("--algorithm") != mapCommandLineArgs.end()) ? mapCommandLineArgs["--algorithm"] : "clique");
    string const sExperimentName((mapCommandLineArgs.find("--experiment") != mapCommandLineArgs.end()) ? mapCommandLineArgs["--experiment"] : "");
    bool   const bPrintHeader(mapCommandLineArgs.find("--header") != mapCommandLineArgs.end());
    string const sTimeout(mapCommandLineArgs.find("--timeout") != mapCommandLineArgs.end() ? mapCommandLineArgs["--timeout"] : "");
    bool   const bRunUnitTests(mapCommandLineArgs.find("--run-tests") != mapCommandLineArgs.end());
    bool   const bNoReduce(mapCommandLineArgs.find("--no-reduce") != mapCommandLineArgs.end());
    size_t const uMaxSelections(mapCommandLineArgs.find("--max-selections") != mapCommandLineArgs.end() ? std::stoi(mapCommandLineArgs["--max-selections"]) : 100000000);
    double const dTargetWeight(mapCommandLineArgs.find("--target-weight") != mapCommandLineArgs.end() ? std::stod(mapCommandLineArgs["--target-weight"]) : numeric_limits<double>::max());
    size_t const uRandomSeed(mapCommandLineArgs.find("--random-seed") != mapCommandLineArgs.end() ? std::stoi(mapCommandLineArgs["--random-seed"]) : 0);
////    size_t const uTimeoutInMilliseconds(mapCommandLineArgs.find("--timeout-in-ms") != mapCommandLineArgs.end() ? std::stoi(mapCommandLineArgs["--timeout-in-ms"]) : 5000)

    cout << "#command :";
    for (int i = 0; i < argc; ++i) {
        cout << " " << argv[i];
    }
    cout << endl << flush;

    double dTimeout(0.0);
    bool   bTimeoutSet(false);
    if (!sTimeout.empty()) {
        try {
            dTimeout = stod(sTimeout);
            bTimeoutSet = true;
        } catch(...) {
            cout << "ERROR!: Invalid --timeout argument, please enter valid double value." << endl << flush;
        }
    }

    srand(uRandomSeed);

    bool   const bRunExperiment(!sExperimentName.empty());
    bool   const bTableMode(bOutputLatex || bOutputTable);

    if (!bTableMode) {
        PrintExperimentalWarning();
#ifdef DEBUG_MESSAGE
        PrintDebugWarning();
#endif //DEBUG_MESSAGE
        if (bRunUnitTests) RunUnitTests();
    }


    if (inputFile.empty()) {
        cout << "ERROR: Missing input file " << endl;
        // ShowUsageMessage();
        // return 1; // TODO/DS
    }

    if (argc <= 1) {
        cout << "usage: " << argv[0] << " --input-file=<filename> [--latex] [--header]" << endl;
    }

    string const name(sAlgorithm);
    Algorithm *pAlgorithm(nullptr);

    int n; // number of vertices
    int m; // 2x number of edges

    vector<list<int>> adjacencyList;
    if (inputFile.find(".graph") != string::npos) {
        if (!bTableMode) cout << "#Reading .graph file format. " << endl << flush;
        adjacencyList = readInGraphAdjListEdgesPerLine(n, m, inputFile);
    } else {
        if (!bTableMode) cout << "#Reading .edges file format. " << endl << flush;
        adjacencyList = readInGraphAdjList(n, m, inputFile);
    }

    vector<vector<int>> adjacencyArray;

    bool const bComputeAdjacencyArray(true);
    if (bComputeAdjacencyArray) {
        adjacencyArray.resize(n);
        for (int i=0; i<n; i++) {
            adjacencyArray[i].resize(adjacencyList[i].size());
            int j = 0;
            for (int const neighbor : adjacencyList[i]) {
                adjacencyArray[i][j++] = neighbor;
            }
        }
        adjacencyList.clear(); // does this free up memory? probably some...
    }

    vector<double> vVertexWeights(adjacencyArray.size(),1);

    if (bWeighted) {
        if (bUseWeightFile) {
            if (!ReadWeights(inputFile, vVertexWeights)) {
                cout << "ERROR! Did not read all vertices in .weights file" << endl << flush;
            }
        } else {
            for (size_t i = 0; i < vVertexWeights.size(); ++i) {
                vVertexWeights[i] = (i+1)%200 + 1;
            }
        }
    }

    if (!solutionFile.empty()) {
        vector<bool> vSolution(adjacencyArray.size(), false);
        ReadSolution(solutionFile, vSolution);
        double dSolutionWeight(0.0);
        for (size_t vertex = 0; vertex < vSolution.size(); ++vertex) {
            if (!vSolution[vertex]) continue;
            dSolutionWeight+=vVertexWeights[vertex];
            for (int const neighbor : adjacencyArray[vertex]) {
                if (vSolution[neighbor]) {
                    cout << "Solution is not an independent set" << endl << flush;
                    exit(1);
                }
            }
        }
        cout << "Solution is valid and has weight " << dSolutionWeight << endl << flush;
        exit(0);
    }

    bool cliqueAlgorithm(true);
    PhasedLocalSearch *pPLS(nullptr);
    if (sAlgorithm=="clique") {
        cliqueAlgorithm = true;
        pPLS = new CliquePhasedLocalSearch(adjacencyArray, vVertexWeights);

        pPLS->SetMaxSelections(uMaxSelections);
        if (bTimeoutSet) pPLS->SetTimeOutInMilliseconds(dTimeout*1000);
        pPLS->SetTargetWeight(dTargetWeight);
        pPLS->SetQuiet(bQuiet);
        pAlgorithm = pPLS;

        bool const bAlgorithmStatus(pAlgorithm->Run());

        // if did not reach target (non-infinite) weight, then it's a failure...
        if (!bAlgorithmStatus && dTargetWeight != numeric_limits<double>::max()) {
            std::cout << "#" << pAlgorithm->GetName() << " reported a failure. Quitting." << std::endl << std::flush;
        }

        // TODO/DS: output run statistics.
        if (!bTableMode) {
            cout << "#OUTPUT        : " << endl << flush;
            cout << "#--------------"   << endl << flush;
            cout << "algorithm-name : " << pAlgorithm->GetName() << endl << flush;
            cout << "git-commit     : " << GIT_COMMIT_STRING << endl << flush;
            cout << "git-status     : " << GIT_STATUS_STRING << endl << flush;
            cout << "graph-name     : " << basename(inputFile) << endl << flush;
            cout << "random-seed    : " << uRandomSeed << endl << flush;

            if (cliqueAlgorithm)
                cout << (bWeighted ? "mwc            : " : "mc             : ");
            else
                cout << (bWeighted ? "mwis           : " : "mis            : ");

            cout << pPLS->GetBestWeight() << endl << flush;

            cout << "target         : " << pPLS->GetTargetWeight() << endl << flush;
            cout << "time(s)        : " << Tools::GetTimeInSeconds(pPLS->GetTimeToBestWeight(), false) << endl << flush;
            cout << "timeout        : " << pPLS->GetTimeoutInSeconds() << endl << flush;
            cout << "penalty-delay  : " << pPLS->GetPenaltyDelay() << endl << flush;
            cout << "selections     : " << pPLS->GetSelectionsToBestWeight() << endl << flush;
            cout << "max-selections : " << pPLS->GetMaxSelections() << endl << flush;
            cout << "best-solution  :";
            for (int const vertex : pPLS->GetBestK()) {
                cout << " " << vertex;
            }
            cout << endl << flush;
        }

    } else if (!bNoReduce) {
        cliqueAlgorithm = false;

        clock_t const startReductions(clock());

        // first remove all isolated vertices, and create new graph
        // and map all nodes, weights
        Isolates4<SparseArraySet> isolates(adjacencyArray, vVertexWeights);

        vector<int> vIndependentSetVertices;
        vIndependentSetVertices.reserve(adjacencyArray.size());
        isolates.RemoveAllIsolates(vIndependentSetVertices);

        double dInitialWeight(0.0);
        // compute the current weight, adjust search weight of remaining problem.
        for (int const vertex : vIndependentSetVertices) {
            dInitialWeight += vVertexWeights[vertex];
        }

        size_t uRemainingGraphSize(isolates.GetInGraph().Size());

        // create new subgraph
        vector<vector<int>> subgraph;
        map<int,int> newToOldVertexMap;
        GraphTools::ComputeInducedSubgraphIsolates(isolates, subgraph, newToOldVertexMap);
        // create new weights
        vector<double> vNewWeights(subgraph.size(), 0.0);
        for (pair<int,int> newToOldVertex : newToOldVertexMap) {
            int const newVertex(newToOldVertex.first);
            int const oldVertex(newToOldVertex.second);
            vNewWeights[newVertex] = vVertexWeights[oldVertex];
        }
        clock_t const endReductions(clock());

        pPLS = new IndependentSetPhasedLocalSearch(subgraph, vNewWeights);

        pPLS->SetMaxSelections(uMaxSelections);
        if (bTimeoutSet) pPLS->SetTimeOutInMilliseconds(dTimeout*1000);
        pPLS->SetTargetWeight(dTargetWeight - dInitialWeight);
        pPLS->SetQuiet(bQuiet);
        pAlgorithm = pPLS;
        bool bAlgorithmStatus(true);


        // if the remaining graph is not empty, continue w/ local search
        if (uRemainingGraphSize != 0) {
            bAlgorithmStatus = pAlgorithm->Run();
        }

        // if did not reach target (non-infinite) weight, then it's a failure...
        if (!bAlgorithmStatus && dTargetWeight != numeric_limits<double>::max()) {
            std::cout << "#" << pAlgorithm->GetName() << " reported a failure. Quitting." << std::endl << std::flush;
        }

        // Take best independent set and remap back to original node set.
        for (int const vertex : pPLS->GetBestK()) {
            vIndependentSetVertices.push_back(newToOldVertexMap[vertex]);
        }

        // TODO/DS: output run statistics.
        if (!bTableMode) {
            cout << "#OUTPUT         : " << endl << flush;
            cout << "#---------------"   << endl << flush;
            cout << "algorithm-name  : " << pAlgorithm->GetName() << endl << flush;
            cout << "git-commit      : " << GIT_COMMIT_STRING << endl << flush;
            cout << "git-status      : " << GIT_STATUS_STRING << endl << flush;
            cout << "graph-name      : " << basename(inputFile) << endl << flush;
            cout << "graph-size      : " << adjacencyArray.size() << endl << flush;
            cout << "reduced-graph   : " << uRemainingGraphSize << endl << flush;
            cout << "reduced-time(s) : " << Tools::GetTimeInSeconds((endReductions - startReductions), false) << endl << flush;
            cout << "random-seed     : " << uRandomSeed << endl << flush;

            if (cliqueAlgorithm)
                cout << (bWeighted ? "mwc             : " : "mc              : ");
            else
                cout << (bWeighted ? "mwis            : " : "mis             : ");

            cout << pPLS->GetBestWeight() + dInitialWeight << endl << flush;

            cout << "target          : " << pPLS->GetTargetWeight() + dInitialWeight << endl << flush;
            cout << "time(s)         : " << Tools::GetTimeInSeconds(pPLS->GetTimeToBestWeight(), false) << endl << flush;
            cout << "timeout         : " << pPLS->GetTimeoutInSeconds() << endl << flush;
            cout << "penalty-delay   : " << pPLS->GetPenaltyDelay() << endl << flush;
            cout << "selections      : " << pPLS->GetSelectionsToBestWeight() << endl << flush;
            cout << "total-selections: " << pPLS->GetSelections() << endl << flush;
            cout << "max-selections  : " << pPLS->GetMaxSelections() << endl << flush;
            cout << "best-solution   :";
            for (int const vertex : vIndependentSetVertices) {
                cout << " " << vertex;
            }
            cout << endl << flush;
        }
    } else {
        cliqueAlgorithm = false;

        pPLS = new IndependentSetPhasedLocalSearch(adjacencyArray, vVertexWeights);

        pPLS->SetMaxSelections(uMaxSelections);
        if (bTimeoutSet) pPLS->SetTimeOutInMilliseconds(dTimeout*1000);
        pPLS->SetTargetWeight(dTargetWeight);
        pPLS->SetQuiet(bQuiet);
        pAlgorithm = pPLS;
        bool const bAlgorithmStatus = pAlgorithm->Run();

        // if did not reach target (non-infinite) weight, then it's a failure...
        if (!bAlgorithmStatus && dTargetWeight != numeric_limits<double>::max()) {
            std::cout << "#" << pAlgorithm->GetName() << " reported a failure. Quitting." << std::endl << std::flush;
        }

        // TODO/DS: output run statistics.
        if (!bTableMode) {
            cout << "#OUTPUT         : " << endl << flush;
            cout << "#---------------"   << endl << flush;
            cout << "algorithm-name  : " << pAlgorithm->GetName() << endl << flush;
            cout << "git-commit      : " << GIT_COMMIT_STRING << endl << flush;
            cout << "git-status      : " << GIT_STATUS_STRING << endl << flush;
            cout << "graph-name      : " << basename(inputFile) << endl << flush;
            cout << "graph-size      : " << adjacencyArray.size() << endl << flush;
            cout << "reductions      : " << (bNoReduce ? "OFF" : "ON") << endl << flush;
            cout << "reduced-graph   : " << adjacencyArray.size() << endl << flush;
            cout << "reduced-time(s) : " << Tools::GetTimeInSeconds(0, false) << endl << flush;
            cout << "random-seed     : " << uRandomSeed << endl << flush;

            if (cliqueAlgorithm)
                cout << (bWeighted ? "mwc             : " : "mc              : ");
            else
                cout << (bWeighted ? "mwis            : " : "mis             : ");

            cout << pPLS->GetBestWeight() << endl << flush;

            cout << "target          : " << pPLS->GetTargetWeight() << endl << flush;
            cout << "time(s)         : " << Tools::GetTimeInSeconds(pPLS->GetTimeToBestWeight(), false) << endl << flush;
            cout << "timeout         : " << pPLS->GetTimeoutInSeconds() << endl << flush;
            cout << "penalty-delay   : " << pPLS->GetPenaltyDelay() << endl << flush;
            cout << "selections      : " << pPLS->GetSelectionsToBestWeight() << endl << flush;
            cout << "total-selections: " << pPLS->GetSelections() << endl << flush;
            cout << "max-selections  : " << pPLS->GetMaxSelections() << endl << flush;
            cout << "best-solution   :";
            for (int const vertex : pPLS->GetBestK()) {
                cout << " " << vertex;
            }
            cout << endl << flush;
        }
    }

    delete pAlgorithm; pAlgorithm = nullptr;

    if (bRunExperiment) {
        return 0;
    }

#ifdef DEBUG_MESSAGE
    PrintDebugWarning();
#endif

    return 0;
}
