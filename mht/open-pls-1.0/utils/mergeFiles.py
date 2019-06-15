import numpy as np
import matplotlib.pyplot as plt
import csv
import sys, os
import operator
import math
from matplotlib.backends.backend_pdf import PdfPages
from operator import itemgetter
import matplotlib.colors as cl


#ilp, tree, decomp_tree, greedy = zip(*f)

cities = {"BE" : "Berlin", "LO":"London", "WA":"Washing.", "NY":"New York","BO":"Boston", "VI":"Vienna","LA":"LA", "RO":"Rome","SE":"Seattle","MO":"Montreal","PA":"Paris", "ZO":"Key","Median":"Median","BA":"Baltimore"
          }
rgb = [[119,183,44],[225,229,0],[224,155,0],[180,130,31],[187,23,23],[190,0,126],[0,163,226],[0,150,130],[25,25,140],[70,100,170],[160,32,240]]

def lighten(color, ratio=0.5): 
          red, green, blue = color 
          return (red + (1.0 - red) * ratio, green + (1.0 - green) * ratio, 
                blue + (1.0 - blue) * ratio) 

#opacity = 0.5
error_config = {'ecolor': '0.3'}

if sys.argv[len(sys.argv)-1] == "SILENT":
	silent = True
else:
	silent = False


def readFile(inputFile):
	f = csv.reader(open(inputFile))
	## READ IN DATA
	i =0
	dataset = ()


	data = ()
	labels = ()
	labelMap = dict()

	for row in f:
		if i == 0:
			labels = row
			for col in range(len(labels)):
				label = labels[col].strip()
				labelMap.update({label:col})
			labelMap.update({"PRESENCE DENSITY":len(labels)})
		else:
			values = ()
			for value in row:
				if value == '':
					values += (-1,)
				else:
					values += (float(value),)

			
			#values += (values[labelMap["NUMBER OF CONFLICTS"]]+values[labelMap["NUMBER OF VISIBILITIES"]],)
			data = data + (values,)		
		i += 1
	return (data,labelMap,labels)


data1, lm1, labels1 = readFile(sys.argv[1])
data2, lm2, labels2 = readFile(sys.argv[2])

print data2

k = int(sys.argv[3])

mergedData = dict()

for tup in data1:
	seed = tup[lm1["SEED"]]
	assert mergedData.has_key(seed) == False
	assert tup[lm1["K"]] == k
	mergedData.update({seed:tup})


output =()

for tup in data2:
	seed =  tup[lm2["SEED"]]
	#print seed 
	if tup[lm2["K"]] == k or -tup[lm2["K"]] == k:
		if mergedData.has_key(seed):
			mergedData[seed]+= tup[2:]
			output += (mergedData[seed],)
		else:			
			print "Could not find seed: ", seed


with open(sys.argv[4], 'wb') as csvfile:
	f = csv.writer(csvfile, delimiter=',',
		                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
	
	labels1 += labels2[2:]
	f.writerow(list(labels1)) 
	for values in output:
		#values = mergedData[key]		
		f.writerow(list(values))



