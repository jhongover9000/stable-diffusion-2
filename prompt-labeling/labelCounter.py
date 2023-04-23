import csv
import os
import sys
import time
from collections import Counter

readFile = open(sys.argv[1],"r")
outFile = open(sys.argv[2],"w")

labels = []
labelCount = []

for line in readFile:
    line = line.strip()
    labels.append(line)

setLabels = set(labels)
c = Counter(labels)

print(c)
outFile.write(str(c))




