import os
import math
import re
from collections import Counter

path = r"/Users/olzhas/work/morphology_research/data/toy/documents/"


def read_file(path):
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    allWords = []

    for line in lines:
        cleaned = re.findall(r'[^,.!?:;]+', line)
        cleanedLine = ' '.join(cleaned)
        words = cleanedLine.split()

        allWords += words

    return allWords


def load_documents(path):
    dfArr = []
    allWordsArr = {}
    documentLengths = {}
    dir_list = os.listdir(path)

    for el in dir_list:
        file_path = os.path.join(path, el)

        allWords = read_file(file_path)


        documentLengths[el] = len(allWords)

        counts = Counter(allWords)
        allWordsArr[el] = counts

        for item in counts:
            dfArr.append(item)

    return dir_list, allWordsArr, documentLengths, dfArr


def compute_df(dfArr):
    dfCounts = Counter(dfArr)
    return dfCounts


def compute_idf(dfCounts, numDocs):
    IDFDict = {}

    for item, count in dfCounts.items():
        IDF = math.log(numDocs/count)
        IDFDict[item] = IDF

    return IDFDict


def print_tfidf(allWordsArr, IDFDict):
    for name, sen in allWordsArr.items():
        print(name)
        for item, TF in sen.items():
            print(item, ": ", TF * IDFDict[item])
        print()


def print_tfidf_details(allWordsArr, documentLengths, IDFDict):
    for name, sen in allWordsArr.items():
        print(name)
        for item, TF in sen.items():
            print("TF: ", TF)
            print("length: ", documentLengths[name])
            print("normalized TF: ", TF/documentLengths[name])
            print("IDF: ", IDFDict[item])
            print("normalized TF-IDF: ", (TF/documentLengths[name]) * IDFDict[item])
            logTF = math.log(1 + TF)
            print("log TF: ", logTF)
            logTFIDF = logTF * IDFDict[item]
            print("log TF-IDF: ", logTFIDF)

        print()


def search_tfidf(allWordsArr, IDFDict):
    searchWords = input("Search words: ").split()

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
    return sortedSearchResults


def compute_avgdl(documentLengths):
    totalLength = 0
    for name, length in documentLengths.items():
        totalLength += length
    return totalLength / len(documentLengths)

def compute_bm25_idf(dfCounts, numDocs):
    bm25IDF = {}
    for item, count in dfCounts.items():
        IDF = math.log((numDocs - count + 0.5) / (count + 0.5))
        bm25IDF[item] = IDF
    return bm25IDF

def search_bm25(allWordsArr, documentLengths, bm25IDF, avgdl):
    searchWords = input("Search words: ").split()
    
    k1 = 1.5
    b = 0.75
    bm25Result = {}
    
    for name, sen in allWordsArr.items():
        wordScore = 0
        for word in searchWords:
            if(word in sen):
                wordScore += bm25IDF[word] * (sen[word] * (k1 + 1)) / (sen[word] + k1 * (1 - b + b * (documentLengths[name] / avgdl)))
            bm25Result[name] = wordScore
    sortedBm25Result = dict(sorted(bm25Result.items(), key=lambda item: item[1], reverse=True))
    return sortedBm25Result

def main():
    dir_list, allWordsArr, documentLengths, dfArr = load_documents(path)

    dfCounts = compute_df(dfArr)
    numDocs = len(dir_list)

    IDFDict = compute_idf(dfCounts, numDocs)

    avgdl = compute_avgdl(documentLengths)
    bm25IDF = compute_bm25_idf(dfCounts, numDocs)

    bm25Result = search_bm25(allWordsArr, documentLengths, bm25IDF, avgdl)
    print(bm25Result)

    # print_tfidf(allWordsArr, IDFDict)
    # print_tfidf_details(allWordsArr, documentLengths, IDFDict)

    # search_tfidf(allWordsArr, IDFDict)
    # print(compute_bm25_idf(dfCounts, numDocs))
    # print(allWordsArr)

if __name__ == "__main__":
    main()