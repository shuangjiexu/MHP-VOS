Install:
    First, make sure you have clang 7.3.0 or g++ 5.3, then execute:
    $ cd open-pls-1.0
    $ make

Running:
    $ ./bin/pls --algorithm=<mc|mwc|mis|mwis> [--weighted] [--use-weight-file] 
                --input-file=<input graph in METIS format> [--timeout=<timeout in seconds>] [--random-seed=<integer random seed>]

Graph Format:
    graph:
        first line: <vertices number, edges number>
        following line: each line means a vertices, with the connected vertices number with it (start from 1)