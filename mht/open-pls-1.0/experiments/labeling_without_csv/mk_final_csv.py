#! /usr/bin/python

import sys 
import re

from collections import defaultdict

def same_length(hash_list):
    first_loop = 1
    length = -1
    for the_key in hash_list.keys():
        if first_loop == 1:
            first_loop = 0
            length = len(hash_list[the_key])
        if length != len(hash_list[the_key]):
            return 0
    return 1

def read_in_data(file_name, hash_list):
    data_file = open(file_name, 'r')

    files_processed = set()
    data_set_id = "(none)"
    last_am = 3
    #read all lines into hash table
    read_header=1
    for line in data_file:
        columns = line.split(",")
        seed = columns[0].strip()
        k    = columns[1].strip()
        am   = columns[2].strip()
        pls_max_score = columns[3].strip()
        pls_max_time  = columns[4].strip()

        if read_header == 1:
            #print "SEED,K,PLS AM1 SCORE,PLS AM2 SCORE,PLS AM3 SCORE,PLS AM1 TIME,PLS AM2 TIME, PLS AM3 TIME"
            hash_list["SEED"].append("SEED")
            hash_list["K"].append("K")
            hash_list["PLS AM1 SCORE"].append("PLS AM1 SCORE")
            hash_list["PLS AM2 SCORE"].append("PLS AM2 SCORE")
            hash_list["PLS AM3 SCORE"].append("PLS AM3 SCORE")
            hash_list["PLS AM1 TIME"].append("PLS AM1 TIME")
            hash_list["PLS AM2 TIME"].append("PLS AM2 TIME")
            hash_list["PLS AM3 TIME"].append("PLS AM3 TIME")
            read_header = 0
            continue

        if data_set_id != seed + k:
            if last_am != 3:
                while last_am < 3:
                    last_am += 1
                    hash_list["PLS AM" + str(last_am) + " SCORE"].append("")
                    hash_list["PLS AM" + str(last_am) + " TIME"].append("")

            if not same_length(hash_list):
                print "ERROR! uneven number or unordered entries in intermediate.csv"
                sys.exit(1)
            # row complete, write it out
            row = []
            row.append(hash_list["SEED"][-1])
            row.append(hash_list["K"][-1])
            row.append(hash_list["PLS AM1 SCORE"][-1])
            row.append(hash_list["PLS AM2 SCORE"][-1])
            row.append(hash_list["PLS AM3 SCORE"][-1])
            row.append(hash_list["PLS AM1 TIME"][-1])
            row.append(hash_list["PLS AM2 TIME"][-1])
            row.append(hash_list["PLS AM3 TIME"][-1])
            print ",".join(row)
            
            data_set_id = seed + k
            if data_set_id in files_processed:
                print "ERROR! intermediate csv lines aren't consecutive w.r.t data sets"
                sys.exit(1)

            hash_list["SEED"].append(seed)
            hash_list["K"].append(k)

            files_processed.add(data_set_id)


        last_am = int(am[-1])
        hash_list["PLS AM" + str(last_am) + " SCORE"].append(pls_max_score)
        hash_list["PLS AM" + str(last_am) + " TIME"].append(pls_max_time)

    # print last row
    while last_am < 3:
        last_am += 1
        hash_list["PLS AM" + str(last_am) + " SCORE"].append("")
        hash_list["PLS AM" + str(last_am) + " TIME"].append("")

    if not same_length(hash_list):
        print "ERROR! uneven number or unordered entries in intermediate.csv"
        sys.exit(1)

    # row complete, write it out
    row = []
    row.append(hash_list["SEED"][-1])
    row.append(hash_list["K"][-1])
    row.append(hash_list["PLS AM1 SCORE"][-1])
    row.append(hash_list["PLS AM2 SCORE"][-1])
    row.append(hash_list["PLS AM3 SCORE"][-1])
    row.append(hash_list["PLS AM1 TIME"][-1])
    row.append(hash_list["PLS AM2 TIME"][-1])
    row.append(hash_list["PLS AM3 TIME"][-1])
    print ",".join(row)



#map file data into hash table
hash_list = defaultdict(list)
read_in_data(sys.argv[1], hash_list)

#reduce each hash table entry
#reduced_data = []
#reduce_data(hash_list, reduced_data)

#output in csv table line
#print ",".join(reduced_data)
