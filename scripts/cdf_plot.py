import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statistics
import argparse, os
import sys, time
import csv
import subprocess

data30 = []
data50 = []
data70 = []

inputName = sys.argv[1]
title = sys.argv[2]

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
# data30_sd = calculate_standard_deviation(data30)
# data50_sd = calculate_standard_deviation(data50)
# data70_sd = calculate_standard_deviation(data70)
p = 1. * np.arange(len(data30)) / (len(data30) - 1)
print(data30_cdf)

plt.plot(data30_cdf, p, 'r-', label = f'CDF of Inference Time at 50 Steps')
plt.plot(data50_cdf, p, 'g+', label = f'CDF of Inference Time at 50 Steps')
plt.plot(data70_cdf, p, 'bo', label = f'CDF of Inference Time at 70 Steps')

fig = plt.figure()

plt.xlabel('Value')
plt.ylabel('CDF')
plt.title(title)

# Show legend and plot
plt.legend()
plt.show()
plt.savefig("/scratch/jhh508/time-calculations/plot_new.png")
print("done.")


