// local includes
#include "Tools.h"
#include "Algorithm.h"

// system includes
#include <cassert>
#include <cstdio>
#include <ctime>
#include <fstream> // ifstream
//#include <csys/resource.h>

#include <list>
#include <set>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>

using namespace std;

/*! \file Tools.cpp

    \brief A collection of useful comparators and print functions

    \author Darren Strash (first name DOT last name AT gmail DOT com)

    \copyright Copyright (c) 2011 Darren Strash. This code is released under the GNU Public License (GPL) 3.0.

    \image html gplv3-127x51.png

    \htmlonly
    <center>
    <a href="gpl-3.0-standalone.html">See GPL 3.0 here</a>
    </center>
    \endhtmlonly
*/

/*! \brief read in a graph from stdin and return an 
           adjacency list, as an array of linked lists
           of integers.

    \param n this will be the number of vertices in the
             graph when this function returns.

    \param m this will be 2x the number of edges in the
             graph when this function returns.

    \return an array of linked lists of integers (adjacency list) 
            representation of the graph
*/

vector<list<int>> readInGraphAdjList(int* n, int* m)
{
    int u, v; // endvertices, to read edges.

    if(scanf("%d", n)!=1)
    {
        fprintf(stderr, "problem with line 1 in input file\n");
        exit(1);
    }

    if(scanf("%d", m)!=1)
    {
        fprintf(stderr, "problem with line 2 in input file\n");
        exit(1);
    }

#ifdef DEBUG
    printf("Number of vertices: %d\n", *n);
    printf("Number of edges: %d\n", *m);
#endif
    
    vector<list<int>> adjList(*n);

    int i = 0;
    while(i < *m)
    {
        if(scanf("%d,%d", &u, &v)!=2)
        {
            printf("problem with line %d in input file\n", i+2);
            exit(1);
        }
        assert(u < *n && u > -1);
        assert(v < *n && v > -1);
        if(u==v)
            printf("%d=%d\n", u, v);
        assert(u != v);

        adjList[u].push_back(v);

        i++;
    }

#ifdef DEBUG
    printArrayOfLinkedLists(adjList, *n);
#endif

    return adjList;
}

vector<list<int>> readInGraphAdjListEdgesPerLine(int &n, int &m, string const &fileName)
{
    ifstream instream(fileName.c_str());

    if (instream.good() && !instream.eof()) {
        string line;
        std::getline(instream, line);
////        cout << "Read Line: " << line << endl << flush;
        while((line.empty() || line[0] == '%') && instream.good() && !instream.eof()) {
            std::getline(instream, line);
        }
        stringstream strm(line);
        strm >> n >> m;
    } else {
        fprintf(stderr, "ERROR: Problem reading number of vertices and edges in file %s\n", fileName.c_str());
        exit(1);
    }

#ifdef DEBUG
    printf("Number of vertices: %d\n", n);
    printf("Number of edges: %d\n", m);
#endif
    
    vector<list<int>> adjList(n);

    int u(-1);
    int i = 0;
    while (i < n) {
        if (!instream.good()  || instream.eof()) {
            fprintf(stderr, "ERROR: Problem reading line %d in file %s\n", i+1, fileName.c_str());
            exit(1);
        }

        string line;
        std::getline(instream, line);
        u = i; // TODO/DS: remove.
        stringstream strm(line);
////        bool debug(true); ////u == 40656 || u == 40653);
////if (debug)        cout << (u+1) << " : " << endl << flush;
////if (debug)        cout << "Read     Line: " << line << endl << flush;
////if (debug)        cout << "Actually Read: ";
        while (!line.empty() && strm.good() && !strm.eof()) {
            int v(-1);
            strm >> v;
            ////if (!strm.good()) break;
////if (debug)            cout << v << " ";
            v--;
////            cout << "(u,v)=" << u << "," << v << endl << flush;
            assert(u < n && u > -1);
            assert(v < n && v > -1);
            if (u==v) {
                fprintf(stderr, "ERROR: Detected loop %d->%d\n", u + 1, v + 1);
                exit(1);
            }

            adjList[u].push_back(v);
////            cout << "pushed back..." << endl << flush;
        }
////if (debug)        cout << endl << flush;

        i++;
    }

////    cout << "Done reading file..." << endl << flush;
#ifdef DEBUG
    printArrayOfLinkedLists(adjList, n);
#endif

    return adjList;
}


vector<list<int>> readInGraphAdjList(int &n, int &m, string const &fileName)
{

    ifstream instream(fileName.c_str());

    if (instream.good() && !instream.eof())
        instream >> n;
    else {
        fprintf(stderr, "problem with line 1 in input file\n");
        exit(1);
    }


    if (instream.good() && !instream.eof())
        instream >> m;
    else {

        fprintf(stderr, "problem with line 2 in input file\n");
        exit(1);
    }

#ifdef DEBUG
    printf("Number of vertices: %d\n", n);
    printf("Number of edges: %d\n", m);
#endif
    
    vector<list<int>> adjList(n);

    int u, v; // endvertices, to read edges.
    int i = 0;
    while(i < m)
    {
        char comma;
        if (instream.good() && !instream.eof()) {
            instream >> u >> comma >> v;
        } else {
            fprintf(stderr, "problem with line %d in input file\n", i+2);
            exit(1);
        }
        assert(u < n && u > -1);
        assert(v < n && v > -1);
        if(u==v)
            fprintf(stderr, "Detected loop %d->%d\n", u, v);
        assert(u != v);

        adjList[u].push_back(v);

        i++;
    }

#ifdef DEBUG
    printArrayOfLinkedLists(adjList, n);
#endif

    return adjList;
}

#if 0
vector<list<int>> readInGraphAdjListDimacs(int &n, int &m, string const &fileName)
{

    std::getline(instream, line);
    ifstream instream(fileName.c_str());

    if (instream.good() && !instream.eof())
        instream >> m;
    else {

        fprintf(stderr, "problem with line 2 in input file\n");
        exit(1);
    }

#ifdef DEBUG
    printf("Number of vertices: %d\n", n);
    printf("Number of edges: %d\n", m);
#endif
    
    vector<list<int>> adjList(n);

    int u, v; // endvertices, to read edges.
    int i = 0;
    while(i < m)
    {
        char comma;
        if (instream.good() && !instream.eof()) {
            instream >> u >> comma >> v;
        } else {
            fprintf(stderr, "problem with line %d in input file\n", i+2);
            exit(1);
        }
        assert(u < n && u > -1);
        assert(v < n && v > -1);
        if(u==v)
            fprintf(stderr, "Detected loop %d->%d\n", u, v);
        assert(u != v);

        adjList[u].push_back(v);

        i++;
    }

#ifdef DEBUG
    printArrayOfLinkedLists(adjList, n);
#endif

    return adjList;
}
#endif

void RunAndPrintStats(Algorithm *pAlgorithm, list<list<int>> &cliques, bool const outputLatex)
{
    fprintf(stderr, "%s: ", pAlgorithm->GetName().c_str());
    fflush(stderr);

    clock_t start = clock();

    pAlgorithm->Run();

    clock_t end = clock();

    if (!outputLatex) {
        fprintf(stderr, "Running time: %f seconds\n", (double)(end-start)/(double)(CLOCKS_PER_SEC));
    } else {
        printf("%.2f", (double)(end-start)/(double)(CLOCKS_PER_SEC));
    }
    fflush(stderr);
}

void InvertGraph(vector<list<int>> const &adjList)
{
    int const n(adjList.size());
    cout << n << endl;
    size_t numEdgesInInverse(0);
    for (list<int> const &neighbors : adjList) {
        numEdgesInInverse += n - neighbors.size() - 1; // all non-edges except loops
    }

    cout << numEdgesInInverse << endl;

    for (int i = 0; i < adjList.size(); ++i) {
        set<int> setNeighbors;
        setNeighbors.insert(adjList[i].begin(), adjList[i].end());
        for (int neighbor=0; neighbor < adjList.size(); neighbor++) {
            if (setNeighbors.find(neighbor) == setNeighbors.end() && neighbor != i) {
                cout << "(" << i << "," << neighbor << i << ")" << endl;
            }
        }
    }
}

string Tools::GetTimeInSeconds(clock_t delta, bool const brackets) {
    stringstream strm;

    strm.precision(4);
    strm.setf(std::ios::fixed, std::ios::floatfield);
    if (brackets) {
        strm << "[" << (double)(delta)/(double)(CLOCKS_PER_SEC) << "s]";
    } else {
        strm << (double)(delta)/(double)(CLOCKS_PER_SEC) << "s";
    }
    return strm.str();
}
