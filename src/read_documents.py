import os
import math
import re
from collections import Counter

path = r"/Users/olzhas/work/morphology_research/data/toy/documents/"

dir_list = os.listdir(path)

print("Files and directories in '", path, "' :")
print(dir_list)

dfArr = []
allWordsArr = {}

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

    counts = Counter(allWords)
    allWordsArr[el] = counts

    for item, count in counts.items():
        dfArr.append(item);

    print("Word frequencies: ")
    print(counts)
    file1.close()
    print()

dfCounts = Counter(dfArr)
numDocs = len(dir_list)
print("DF: ")

for item, count in dfCounts.items():
    print(f"{item}: {count}")

print()

IDFDict = {}
print("IDF: ")

for item, count in dfCounts.items():
    IDF = math.log(numDocs/count)
    IDFDict[item] = IDF
    print(f"{item}: {IDF}")

for name, sen in allWordsArr.items():
    print(name)
    for item, TF in sen.items():
        print(item, ": ", TF * IDFDict[item])
    print()

searchWords = input("Search word: ").split()

searchResults = {}

for name, sen in allWordsArr.items():
    sumRes = 0
    for item, TF in sen.items():
        TFIDF = TF * IDFDict[item]
        for st in searchWords:
            if(st == item):
                sumRes += TFIDF
    searchResults[name] = sumRes

sortedSearchResults = dict(sorted(searchResults.items(), key=lambda item: item[1], reverse=True))
print(sortedSearchResults)