import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statistics
import argparse, os
import sys, time
import csv
import subprocess
import pandas as pd
from itertools import accumulate

data30 = []
data50 = []
data70 = []

outdir = sys.argv[1]
inputName = sys.argv[2]
# "out"
try:
    opt = sys.argv[3]
except:
    opt = ""

def calculate_standard_deviation(data):
    return statistics.stdev(data)

inputFile = open(inputName, 'r')

for line in inputFile:
    line = line.strip().split(",")
    data30.append(float(line[0]))
    data50.append(float(line[1]))
    data70.append(float(line[2]))

data30_cdf = np.sort(data30)
data50_cdf = np.sort(data50)
data70_cdf = np.sort(data70)

# GENERATE EXAMPLE DATA
df = pd.DataFrame()
df['30 Steps'] = data30_cdf
df['50 Steps'] = data50_cdf
df['70 Steps'] = data70_cdf

# START A PLOT
fig,ax = plt.subplots()

for col in df.columns:

  # SKIP IF IT HAS ANY INFINITE VALUES
  if not all(np.isfinite(df[col].values)):
    continue

  # USE numpy's HISTOGRAM FUNCTION TO COMPUTE BINS
  xh, xb = np.histogram(df[col], bins=60, normed=True)

  # COMPUTE THE CUMULATIVE SUM WITH accumulate
  xh = list(accumulate(xh))
  # NORMALIZE THE RESULT
  xh = np.array(xh) / max(xh)

  # PLOT WITH LABEL
  ax.plot(xb[1:], xh, label=f"{col}")
ax.legend()
plt.title("CDFs of SD v2.0 Inference Times Taken With" + opt + " Neg Prompts")
plt.show()
plt.savefig("/scratch/jhh508/time-calculations/plot_new" + str(len(os.listdir(outdir))) +".png")
print("done.")