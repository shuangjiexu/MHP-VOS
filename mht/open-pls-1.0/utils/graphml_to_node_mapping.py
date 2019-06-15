#!  /bin/python

import sys
import re
from collections import defaultdict

invert = 0

node_map = defaultdict(int)

def read_graphml(filename, node_map):
    v = -1
    id_for_v = "DEADBEEF"

    id_key_pattern = re.compile("""<key id="(key\d+)" for="node" attr.name="id" """)
    vis_key_pattern = re.compile("""<key id="(key\d+)" for="node" attr.name="visibility" """)
    node_pattern = re.compile("""node id="n(\d+)">""")
    id_pattern = node_pattern #just to create variable, not used yet.
    vis_pattern = node_pattern #just to create variable, not used yet.

    found_key = 0
    id_key = ""
    vis_key = ""

    with open(filename) as f:
        for line in f:
            if not found_key==2:
                key_match = re.search(id_key_pattern, line)
                vis_match = re.search(vis_key_pattern, line)
                if key_match:
                    id_key = key_match.group(1)
                    found_key = 1
                    #print "Found id key = " + id_key
                    id_pattern = re.compile("<data key=\"" + id_key + "\">(.*)</data>")
                if vis_match:
                    vis_key = vis_match.group(1)
                    found_key = 2
                    vis_pattern = re.compile("<data key=\"" + vis_key + "\">\[(.*) / (.*)\]</data>")
            else:
                # done processing nodes
                if "<edge" in line:
                    break
                node_match = re.search(node_pattern, line)
                if node_match:
                    v = int(node_match.group(1))

                id_match = re.search(id_pattern, line)
                if id_match:
                    id_for_v = id_match.group(1)
                vis_match = re.search(vis_pattern, line)

                if vis_match:
                    vis_coord_1 = vis_match.group(1)
                    vis_coord_2 = vis_match.group(2)
                    the_value = (vis_coord_1,vis_coord_2,id_for_v)
                    #print the_key, "->", v
                    node_map[v] = the_value

if len(sys.argv) < 1:
    print "usage: graphml_to_node_mapping <graphml file>"
    sys.exit(1)

# read graphml file, map large ids to {0,...,n} node ids
read_graphml(sys.argv[1], node_map)

for the_key in node_map.keys():
    value = node_map[the_key]
    print str(the_key), str(value[0]) + "," + str(value[1]) + ",\"{" + str(value[2]) + "}\""

# read solution csv file, write out mapped nodes.
#process_csv(sys.argv[2], node_map)
