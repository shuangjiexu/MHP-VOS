#!  /bin/python

import sys
import re
from collections import defaultdict

invert = 0

neighbors = defaultdict(list)

vertices = 0
edges = 0

v = -1
  
node_pattern = re.compile("""node id="n(\d+)">""")
edge_pattern = re.compile("""source="n(\d+)" target="n(\d+)""")
for line in sys.stdin:
    node_match = re.search(node_pattern, line)
    if node_match:
        v = int(node_match.group(1))
        vertices = max(vertices, v+1)

    edge_match = re.search(edge_pattern, line)
    if edge_match:
        v1 = int(edge_match.group(1))
        v2 = int(edge_match.group(2))
#       print "Found edge", v1, "<->", v2

        neighbors[v1].append(v2)
        neighbors[v2].append(v1)
        edges = edges + 1

if vertices == 0:
    print "ERROR!: no vertices found"
    sys.exit(1)

print vertices,edges

for i in range(0,vertices):
    if len(neighbors[i]) == 0:
        print ""
        continue
    neighbors[i].sort()
    previous = -1
    for j in range(0, len(neighbors[i])):
        if neighbors[i][j]==previous:
            continue
        previous = neighbors[i][j]
        if j != 0:
            sys.stdout.write(" ")
        sys.stdout.write(str(neighbors[i][j] + 1));
    sys.stdout.write('\n')
