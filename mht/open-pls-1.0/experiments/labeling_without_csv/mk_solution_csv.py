#! /usr/bin/python

import sys 
import re

from collections import defaultdict

def read_in_data(file_name, hash_list):
    data_file = open(file_name, 'r')

    #read all lines into hash table
    for line in data_file:
        if line.startswith("#") or len(line)==0:
            continue
        #print line.strip()
        key_value_pair = line.split(":")

        key   = key_value_pair[0].strip()
        value = key_value_pair[1].strip()
        #print key + "=" + value
        hash_list[key].append(value)

def all_same(data_array):
    first_element=1
    first=1
    for i in data_array:
        if first==1:
            first_element = i
            first==0
        if first_element != i:
            return 0
    return 1

def all_different(data_array):
    first=1
    value_hash=set()
    for i in data_array:
        value_hash.add(i)

    return len(value_hash) == len(data_array)
            
def same_element_count(hash_list):
    count = 0
    first_key = 1
    for key in hash_list.keys():
        if first_key == 1:
            first_key = 0
            count = len(hash_list[key])
        if count != len(hash_list[key]):
            return 0
    return 1

def reduce_data(hash_list, reduced_array):

    if not same_element_count(hash_list):
        print "ERROR! not all runs have same number of statistics"
        sys.exit(1)

    if not all_same("git-commit"):
        print "ERROR! different git commits in experiment!"
        sys.exit(1)

#    if not all_same("git-status") or len(hash_list["git-status"][0]) != 0:
#        print "ERROR! dirty git status in experiment!"
#        sys.exit(1)

    #data set name first
    if not all_same("graph-name"):
        print "ERROR! different graph names!"
        sys.exit(1)

    #reduced_array.append("\\verb|" + hash_list["graph-name"][0].replace("-sorted", "") + "|")
    #reduced_array.append(hash_list["file-seed"][0])
    #reduced_array.append(hash_list["file-k"][0])
    #reduced_array.append(hash_list["file-am"][0])

    if not all_different(hash_list["random-seed"]):
        print "ERROR! duplicate random seeds!"
        sys.exit(1)

    if not all_same("target"):
        print "ERROR! different target size!"
        sys.exit(1)

    if not all_same("max-selections"):
        print "ERROR! different max selections!"
        sys.exit(1)

    if not all_same("timeout"):
        print "ERROR! different timeouts!"
        sys.exit(1)

    #avg mwis size
    #reduced_array.append("{0:.2f}".format(sum([float(x) for x in hash_list["mwis"]])/len(hash_list["mwis"])))

    #max mwis size
    #max_mwis = max([float(x) for x in hash_list["mwis"]])
    #max_mwis_string = str("{0:.2f}".format(max_mwis))
    #target = float(hash_list["target"][0])

    #reduced_array.append(max_mwis_string)

    #number achieving max mwis size
    #reduced_array.append(str(hash_list["mwis"].count(max_mwis_string)))

    # maximum number of selections
    #reduced_array.append(hash_list["max-selections"][0])

    #average selections
    #reduced_array.append("{0:.2f}".format(sum([float(x) for x in hash_list["selections"]])/len(hash_list["selections"])))

    # timeout
    #reduced_array.append(hash_list["timeout"][0])
    
    #average time
    #reduced_array.append("{0:.2f}".format(sum([float(x.strip("s")) for x in hash_list["time(s)"]])/len(hash_list["time(s)"])))

    #average penalty delay 
    #reduced_array.append("{0:.2f}".format(sum([float(x) for x in hash_list["penalty-delay"]])/len(hash_list["penalty-delay"])))

    #last git commit tag
    #reduced_array.append("..." + hash_list["git-commit"][0][32:])

    #last git status
    #git_status_string = ("Clean" if (len(hash_list["git-status"][0])==0) else "Dirty")
    #reduced_array.append(git_status_string)

def get_best_solution(hash_list):
    best_index = -1
    best_mwis = -1
    for i in range(0,len(hash_list["mwis"])):
        new_mwis = float(hash_list["mwis"][i])
        if new_mwis > best_mwis:
            best_index = i
            best_mwis = new_mwis

    solution_string = hash_list["best-solution"][best_index]
    solution = [int(x) for x in solution_string.split(" ")]
    #print "solution=", solution
    return solution

def read_in_node_mapping(file_name, node_mapping):
    data_file = open(file_name, 'r')

    #read all lines into hash table
    for line in data_file:
        node_string,mapped_string = line.split(" ")
        node_mapping[int(node_string)] = mapped_string.strip()
        #print int(node_string), "->", mapped_string 


def write_mapped_solution(solution, node_mapping):
    for vertex in solution:
        if not vertex in node_mapping.keys():
            print "ERROR! could not find vertex ", vertex, " in node mapping"
            sys.exit(1)
        print node_mapping[vertex]

#map file data into hash table
hash_list = defaultdict(list)
read_in_data(sys.argv[1], hash_list)

solution = get_best_solution(hash_list)

node_mapping = defaultdict(str)
read_in_node_mapping(sys.argv[2], node_mapping)

write_mapped_solution(solution, node_mapping)

#reduce each hash table entry
#reduced_data = []
#reduce_data(hash_list, reduced_data)

#output in csv table line
#print ",".join(reduced_data)
