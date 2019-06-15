#!  /bin/python

import sys
import re
from collections import defaultdict

invert = 0

neighbors = defaultdict(list)

v = -1

weight_key_pattern = re.compile("""<key id="(key\d+)" for="node" attr.name="weight" """)

found_key = 0
weight_key = ""

node_pattern = re.compile("""node id="n(\d+)">""")
weight_pattern = node_pattern
for line in sys.stdin:
    #print "Read: " + line

    if not found_key:
        key_match = re.search(weight_key_pattern, line)
        if key_match:
            weight_key = key_match.group(1)
            found_key = 1
            #print "Found weight key = " + weight_key
            weight_pattern = re.compile("<data key=\"" + weight_key + "\">(.*)</data>")
    else:
        # done evaluating nodes.
        if "<edge" in line:
            break
        node_match = re.search(node_pattern, line)
        if node_match:
            v = int(node_match.group(1))

        weight_match = re.search(weight_pattern, line)
        if weight_match:
            weight = float(weight_match.group(1))
            print(v,weight)
