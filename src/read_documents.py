import os
import re
from collections import Counter

path = r"/Users/olzhas/work/morphology_research/data/toy/documents/"

dir_list = os.listdir(path)

print("Files and directories in '", path, "' :")
print(dir_list)

for el in dir_list:
    file1 = open(path+el, 'r')
    Lines = file1.readlines()
    print(el)

    allWords = [];

    for line in Lines:

        cleaned = re.findall(r'[^,.!?:;]+', line)
        cleanedLine = ' '.join(cleaned)

        print(cleanedLine)

        words = cleanedLine.split()
        num = len(words)

        print("Numbers of words: ", num)
        print("Words: ", words)
        allWords += words

    print("Word frequencies: ")
    print(Counter(allWords))
    file1.close()
    print()