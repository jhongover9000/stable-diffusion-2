import os
import openai
import argparse
import sys
import time
import csv
openai.api_key = os.getenv("OPENAI_API_KEY=key")

readFile = open(sys.argv[1],"r")

# SORT LIST
# promptList = []
# count = 0

# for lines in readFile:
#     # link, imagename, prompt
#     line = lines.strip().split(",")
#     print(line)
#     if(len(line) < 3):
#         continue
#     # we want the prompt
#     promptList.append(line[2].strip())
#     count+=1
# print("Total prompts: " + str(len(promptList)))

# readFile.close()

# if len(promptList) == count:
#     outFile = open(sys.argv[2], "w")

#     for prompt in promptList:
#         outFile.write(prompt + "\n")

#     outFile.close()

#     exit()

# else:
#     print("Mismatach: " + str(len(promptList)) + " | " + str(count))
#     exit()





# GET LABELS
promptStarter = "Label the prompt as one among the following: food, landscape, object, animal, celebrity, person/people, sports, text. Do not use any other words. Prompt: "
openai.api_key = "key"
# ended at 483 + 171

outFile = open("label_list_new.txt", "w")
# testFlag = True
print("Setup done, starting read.")

startLine = 0
labels = []
labelCount = 0
lineCount = 1

for prompt in readFile:

    lineCount+=1
    if(lineCount < startLine):
        labelCount+=1
        continue

    # print(prompt)

    # if(testFlag):
    #     # in case something doesnt work
    #     testVar = input("Check: ")
    #     # if things are OK, proceed unsupervised
    #     if (testVar == "OK"):
    #         testFlag = False

    promptFull = promptStarter + prompt

    # in case the engine is overloaded
    delay = 1
    for n in range(15):
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": promptFull}
                ]
            )
        except:
            # increment each time error occurs
            time.sleep(delay)
            delay *= 2
            continue
        break
        
    label = completion.choices[0].message.content
    print(label.lower().strip("."))
    labels.append(label.lower().strip("."))
    labelCount += 1
    print(labelCount)
    outFile.write(label.lower().strip(".") + "\n")

outFile.close()
    #check 484 for double label, + 171