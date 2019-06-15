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

f = csv.reader(open(str(sys.argv[1])))
## READ IN DATA
i =0
dataset = ()

colors=list(tuple(float(x)/float(255) for x in tup) for tup in rgb)

data = ()

labelMap = dict()

for row in f:
	if i == 0:
		labels = row
		for col in range(len(labels)):
			label = labels[col].strip()
			labelMap.update({label:col})
		labelMap.update({"PRESENCE DENSITY":len(labels)})
	else:
		values = tuple(map(float, row))
		values += (values[labelMap["NUMBER OF CONFLICTS"]]+values[labelMap["NUMBER OF VISIBILITIES"]],)
		data = data + (values,)		
	i += 1




	


sortedData =  sorted(data,key=lambda x: x[labelMap['PRESENCE DENSITY']])


plt.figure(figsize=(3.8,1.7))


def median(mylist):
    sorts = sorted(mylist)
    length = len(sorts)
    if not length % 2:
        return ((sorts[length / 2] + sorts[length / 2 - 1])) / 2.0
    return sorts[length / 2]

offsetX = 10

def plot(plt, data, key, key2, bound,gap,t,color,title,execute):
	yvalues = ()
	xvalues = ()
	length = ()
        i=offsetX
	for tup in data:
		ref = tup[labelMap[key2]]
		if ref <= 0 and tup[labelMap[bound]] > 0 and tup[labelMap[bound]] < 100000000 and tup[labelMap[gap]] <= 5:
			ref = tup[labelMap[bound]] 

		if ref > 0:
			yvalue = tup[labelMap[key]] 
			if yvalue < 0: 
				yvalue=0
				i+=1
			else:	
				yvalues = yvalues + (yvalue/ref*100,)
				xvalues += (i,)	
				length += ((i,tup[labelMap["PRESENCE DENSITY"]]),)			
				i += 1		
	if execute == True:
		plt.plot(xvalues,yvalues,t,markersize=4,c=lighten(color,0.4),zorder=2)
        med    = median(yvalues)
	minVal = min(yvalues)
	avg    = sum(yvalues)/len(yvalues)
	

	return (((minVal,"Min. "+title,color),(med,"Median "+title,color),(avg,"Avg. "+title,color)),min(xvalues),max(xvalues), length)




algo1 = "GREEDY AM* SCORE"
algo2 = "PLS AM* SCORE"

opt = "ILP AM* SCORE"
alt = "ILP AM* BOUND"
gap = "ILP AM* GAP"



model = str(sys.argv[3])

algo1 = algo1.replace("*",str(model))
algo2 = algo2.replace("*",str(model))
opt = opt.replace("*",str(model))
alt = alt.replace("*",str(model))
gap = gap.replace("*",str(model))




if sys.argv[4] == "gmt":
	re1 = plot(plt,sortedData,algo1,opt,alt,gap,"o",colors[9],"Greedy",True)
	re2 = plot(plt,sortedData,algo2,opt,alt,gap,"s",colors[4],"PLS",True)
else:
	re1 = plot(plt,sortedData,"GREEDY AM1 SCORE",opt,alt,gap,"o",colors[9],"Greedy",True)
	re2 = plot(plt,sortedData,"PLS AM1 SCORE",opt,alt,gap,"s",colors[4],"PLS",True)
def printStat(stat):
	print "min=",stat[0][0]
	print "med=",stat[0][1]
	print "avg=",stat[0][2]

print "Greedy"
printStat(re1)

print "PLS"
printStat(re2)

stat  =  sorted(re1[0] + re2[0])


maxY = max(stat)
minY = min(stat)

ax1 = plt.gca()



#ax = plt.gca()

inv = ax1.transData.inverted()


rangeY =  ax1.transData.transform(np.array([(0,minY[0]),(0,maxY[0])]))
offset = (rangeY[1][1]-rangeY[0][1])/(len(stat)-1)


def hLength(pixels):
	origin = inv.transform(np.array([(0,0)]))
    	return abs(origin[0][0]-inv.transform(np.array([(pixels,0)]))[0][0])


left  = min(re1[1],re2[1])
right = max(re1[2],re2[2])+hLength(10)
length = hLength(85)
lineStart = hLength(20) +right
textStart = hLength(5) + lineStart
fs = 6


def plotLines(w,f,text):
	for i in range(len(stat)):
		tup = stat[i]
		
		plt.plot((0,right),(tup[0],tup[0]),c=lighten(tup[2],f),lw=w,zorder=1)    	

		if i == 0 or i == len(stat)-1:
			y = tup[0]
		else:
			label = ax1.transData.transform(np.array([(0,tup[0])]))
			y = rangeY[0][1]+offset*i
			y = inv.transform(np.array([(0,y)]))[0][1]
			

		if text == True:			
			plt.text(textStart,y,tup[1],{'ha': 'left', 'va': 'bottom','fontsize':fs,'color':tup[2]})
		ax1.plot((right,lineStart),(tup[0],y),c=lighten(tup[2],f),lw=w,zorder=1)
		ax1.plot((lineStart,lineStart+length),(y,y),c=lighten(tup[2],f),lw=w,zorder=1)


plotLines(1,0.5,False)
plotLines(0.5,0.0,True)


	


plt.tick_params(labelsize=fs)
plt.tight_layout()
plt.xlim(left,lineStart+length)
ax = plt.gca()
ax.set_ylim(top=102)

def amax(tup):
	best = ((0,0),)
	for el in tup:
		if max(el, key=lambda x:x[1])[1] > max(best,key=lambda x:x[1])[1]:
			best = el
	return best


length = amax((re1[3],re2[3]))#max(re1[3],re2[3],re3[3])	

def toGeqString(number):
	return "${\geq"+str(int(number)) +"}$"

plt.xticks((length[0][0],length[len(length)/4][0],length[len(length)/2][0],length[len(length)*3/4][0],length[len(length)-1][0]), 
          (toGeqString(length[0][1]),
           toGeqString(length[len(length)/4][1]),
           toGeqString(length[len(length)/2][1]),
           toGeqString(length[len(length)*3/4][1]),
           toGeqString(length[len(length)-1][1])),size=fs)



def formatLabels(ax):
	labels = [item.get_text() for item in ax.get_yticklabels()]
	for i in range(len(labels)):
		if len(labels[i])>0:
			labels[i] = "$"+labels[i]+"$"	
	ax.set_yticklabels(labels)


formatLabels(ax)


pp = PdfPages(str(sys.argv[2]))
plt.savefig(pp, format='pdf', bbox_inches='tight', pad_inches = 0)
pp.close()

if silent == False:
	plt.show()



