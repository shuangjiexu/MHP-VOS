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
                    the_key = (vis_coord_1,vis_coord_2,id_for_v)
                    #print the_key, "->", v
                    node_map[the_key] = v

def process_csv(filename, node_map):

    with open(filename) as f:
        for line in f:
            vis_and_id = line.split(",")
            vis_coord_1 = vis_and_id[0]
            vis_coord_2 = vis_and_id[1]
            id_for_v    = vis_and_id[2][2:-4]
            the_key = (vis_coord_1, vis_coord_2, id_for_v)
            #print "querying", the_key
            if (the_key in node_map.keys()):
                print node_map[the_key]
            #else:
                #print "key ", the_key, " not found."

if len(sys.argv) < 3:
    print "usage: mk_solution_file <graphml file> <solution csv file>"
    sys.exit(1)

# read graphml file, map large ids to {0,...,n} node ids
read_graphml(sys.argv[1], node_map)

# read solution csv file, write out mapped nodes.
process_csv(sys.argv[2], node_map)
