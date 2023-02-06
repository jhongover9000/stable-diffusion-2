import argparse, os
import sys, time
import csv
import subprocess

mainDir = sys.argv[1]

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

def scrapeData(file):
    readFile = open(os.path.join(subdir, file), "r")
    # skip prompt
    readFile.readline()
    readFile.readline()
    for i in range(3):
        line = readFile.readline()
        line = line.split(",")
        print(line)
        timeVal = float(line[1])
        if i == 0:
            sum_30 += int(line[1])
            if(timeVal < min_30):
                min_30 = timeVal
            if(timeVal > max_30):
                min_30 = timeVal
            count_30 += 1
        elif i == 1:
            sum_50 += int(line[1])
            if(timeVal < min_50):
                min_50 = timeVal
            if(timeVal > max_50):
                min_50 = timeVal
            count_50 +=1

        elif i == 2:
            sum_70 += int(line[1])
            if(timeVal < min_70):
                min_70 = timeVal
            if(timeVal > max_70):
                min_70 = timeVal
            count_70 += 1
        print("Min 30: " + str(min_30) + "  Max 30: " + str(max_30) + " Avg 30: " + str(sum_30/count_30) + "\n")
        print("Min 50: " + str(min_50) + "  Max 50: " + str(max_50) + " Avg 50: " + str(sum_50/count_50) + "\n")
        print("Min 70: " + str(min_70) + "  Max 70: " + str(max_70) + " Avg 70: " + str(sum_70/count_70) + "\n")


# Iterate through folders in directory
for subdir, dirs, files in os.walk(mainDir):
    for file in files:
        if str(file) == "log.txt":
            scrapeData(file)

print("Done." + "\n")
print("Min 30: " + str(min_30) + "  Max 30: " + str(max_30) + " Avg 30: " + str(sum_30/count_30) + "\n")
print("Min 50: " + str(min_50) + "  Max 50: " + str(max_50) + " Avg 50: " + str(sum_50/count_50) + "\n")
print("Min 70: " + str(min_70) + "  Max 70: " + str(max_70) + " Avg 70: " + str(sum_70/count_70) + "\n")

