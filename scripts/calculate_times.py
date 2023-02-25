# ex: python scripts/calculate_times.py /scratch/jhh508/ timecalc_neg 1 /scratch/jhh508/web-diffusion-neg/

import argparse, os
import sys, time
import csv
import subprocess

mainDir = sys.argv[3]
# 1 for new format, 0 for old
dirType = sys.argv[2]
# output text name
outputName = sys.argv[1]

# Values for 30 steps
sum_30 = 0.0
avg_30 = 0.0
max_30 = 0.0
min_30 = 10.0
count_30 = 0
# Values for 50 steps
sum_50 = 0.0
avg_50 = 0.0
max_50 = 0.0
min_50 = 10.0
count_50 = 0
# Values for 70 steps
sum_70 = 0.0
avg_70 = 0.0
max_70 = 0.0
min_70 = 10.0
count_70 = 0

step30list = []
step50list = []
step70list = []

outdir = "/scratch/jhh508/time-calculations/"

os.makedirs(outdir, exist_ok=True)

# Iterate through folders in directory
for subdir, dirs, files in os.walk(mainDir):
    for file in files:
        if str(file) == "log.txt":
            if(dirType):
                # print(os.path.join(subdir, file))
                readFile = open(os.path.join(subdir, file), "r")
                # skip prompt
                readFile.readline()
                readFile.readline()
                for i in range(3):
                    line = readFile.readline()
                    line = line.strip().split(",")
                    # print(line)
                    try:
                        timeVal = float(line[1])
                    except:
                        print("error on " + os.path.join(subdir, file))
                        print(line)
                        continue

                    if i == 0:
                        sum_30 += timeVal
                        if(timeVal < min_30):
                            min_30 = timeVal
                        if(timeVal > max_30):
                            max_30 = timeVal
                        count_30 += 1
                        step30list.append(timeVal)
                    elif i == 1:
                        sum_50 += timeVal
                        if(timeVal < min_50):
                            min_50 = timeVal
                        if(timeVal > max_50):
                            max_50 = timeVal
                        count_50 +=1
                        step50list.append(timeVal)

                    elif i == 2:
                        sum_70 += timeVal
                        if(timeVal < min_70):
                            min_70 = timeVal
                        if(timeVal > max_70):
                            max_70 = timeVal
                        count_70 += 1
                        step70list.append(timeVal)

                readFile.close()
            else:
                print(os.path.join(subdir, file))
                readFile = open(os.path.join(subdir, file), "r")
                line = readFile.readline()
                line = line.strip().split(",")
                # print(line)
                try:
                    timeVal30 = float(line[2])
                    timeVal50 = float(line[3])
                    timeVal70 = float(line[4])
                except:
                    print("error on " + os.path.join(subdir, file))
                    print(line)
                    continue

                sum_30 += timeVal30
                if(timeVal30 < min_30):
                    min_30 = timeVal30
                if(timeVal30 > max_30):
                    max_30 = timeVal30
                count_30 += 1

                sum_50 += timeVal50
                if(timeVal50 < min_50):
                    min_50 = timeVal50
                if(timeVal50 > max_50):
                    max_50 = timeVal50
                count_50 +=1

                sum_70 += timeVal70
                if(timeVal70 < min_70):
                    min_70 = timeVal70
                if(timeVal70 > max_70):
                    max_70 = timeVal70
                count_70 += 1
                
                readFile.close()

base_count = len(os.listdir(outdir))
outputFile = open(os.path.join(outdir,outputName) + str(base_count), 'w')
for i in range(len(step30list)):
    try:
        outputFile.write(str(step30list[i]) + "," + str(step50list[i]) + "," + str(step70list[i]) + "\n")
    except:
        print("error: " + str(i))
        continue


print("Done." + "\n")
print("Min 30: " + str(min_30) + "  Max 30: " + str(max_30) + " Avg 30: " + str(sum_30/count_30) + " for " + str(count_30) + "images"+ "\n")
print("Min 50: " + str(min_50) + "  Max 50: " + str(max_50) + " Avg 30: " + str(sum_50/count_50) + " for " + str(count_50) + "images"+  "\n")
print("Min 70: " + str(min_70) + "  Max 70: " + str(max_70) + " Avg 30: " + str(sum_70/count_70) + " for " + str(count_70) + "images"+  "\n")

