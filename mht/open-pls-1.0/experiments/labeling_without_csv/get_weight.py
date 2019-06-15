#! /usr/bin/python

import sys 
import re

from collections import defaultdict

am=sys.argv[1]

columns=[]
index=-1
for line in sys.stdin:
    columns = line.split(",")
    if index != -1:
        print columns[index]
        sys.exit(0)
    #print "looking for: ILP " + am + " BOUND"
    for i in range(0,len(columns)):
        if columns[i]==("ILP " + am + " SCORE"):
            #print "found at index ", i
            index = i
