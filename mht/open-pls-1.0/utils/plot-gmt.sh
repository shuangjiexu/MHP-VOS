#! /bin/bash

python mergeFiles.py ./gmt.csv $1 -1 merged-gmt-tmp.csv
python createPLSPlot.py ./merged-gmt-tmp.csv ./output.pdf $2 gmt
