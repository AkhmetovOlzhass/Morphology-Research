import os
import math
import re
from collections import Counter

from evaluation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    mean_metric_at_k,
    ndcg_at_k
)

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


def search_tfidf(allWordsArr, IDFDict, query):
    searchWords = query.split()

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

def search_bm25(allWordsArr, documentLengths, bm25IDF, avgdl, query):
    searchWords = query.split()
    
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
    rankingsBM = {}
    rankingsTFIDF = {}

    queries = {
        "q001": "Казахстан экспорт",
        "q002": "Алматы технологии",
        "q003": "Казахстан нефть",
        "q004": "международный турнир",
    }   

    qrels = {
        "q001": {
            "001.txt": 0,
            "002.txt": 0,
            "003.txt": 1,
            "004.txt": 0,
            "005.txt": 0,
            "006.txt": 0,
            "007.txt": 0,
            "008.txt": 0,
            "009.txt": 0,
            "010.txt": 0,
            "011.txt": 0,
            "012.txt": 1,
            "013.txt": 0,
            "014.txt": 0,
            "015.txt": 0,
            "016.txt": 0,
            "017.txt": 0,
            "018.txt": 0,
            "019.txt": 0,
            "020.txt": 0,
            "021.txt": 0,
            "022.txt": 0,
            "023.txt": 0,
            "024.txt": 0,
            "025.txt": 0,
            "026.txt": 0,
            "027.txt": 0,
            "028.txt": 0,
            "029.txt": 0,
            "030.txt": 0,
            "031.txt": 1,
            "032.txt": 1,
            "033.txt": 0,
            "034.txt": 0,
        },
        "q002": {
            "001.txt": 0,
            "002.txt": 1,
            "003.txt": 0,
            "004.txt": 0,
            "005.txt": 0,
            "006.txt": 0,
            "007.txt": 0,
            "008.txt": 0,
            "009.txt": 0,
            "010.txt": 0,
            "011.txt": 0,
            "012.txt": 0,
            "013.txt": 0,
            "014.txt": 0,
            "015.txt": 1,
            "016.txt": 0,
            "017.txt": 0,
            "018.txt": 0,
            "019.txt": 0,
            "020.txt": 0,
            "021.txt": 0,
            "022.txt": 0,
            "023.txt": 0,
            "024.txt": 0,
            "025.txt": 0,
            "026.txt": 0,
            "027.txt": 0,
            "028.txt": 0,
            "029.txt": 0,
            "030.txt": 0,
            "031.txt": 0,
            "032.txt": 0,
            "033.txt": 0,
            "034.txt": 0,
        },
        "q003": {
            "001.txt": 0,
            "002.txt": 0,
            "003.txt": 0,
            "004.txt": 0,
            "005.txt": 0,
            "006.txt": 0,
            "007.txt": 0,
            "008.txt": 0,
            "009.txt": 0,
            "010.txt": 0,
            "011.txt": 0,
            "012.txt": 1,
            "013.txt": 0,
            "014.txt": 0,
            "015.txt": 0,
            "016.txt": 0,
            "017.txt": 0,
            "018.txt": 0,
            "019.txt": 0,
            "020.txt": 0,
            "021.txt": 0,
            "022.txt": 0,
            "023.txt": 0,
            "024.txt": 0,
            "025.txt": 0,
            "026.txt": 0,
            "027.txt": 0,
            "028.txt": 0,
            "029.txt": 0,
            "030.txt": 0,
            "031.txt": 0,
            "032.txt": 0,
            "033.txt": 0,
            "034.txt": 0,
        },
        "q004": {
            "001.txt": 0,
            "002.txt": 0,
            "003.txt": 0,
            "004.txt": 0,
            "005.txt": 0,
            "006.txt": 0,
            "007.txt": 0,
            "008.txt": 0,
            "009.txt": 1,
            "010.txt": 0,
            "011.txt": 0,
            "012.txt": 0,
            "013.txt": 0,
            "014.txt": 0,
            "015.txt": 0,
            "016.txt": 0,
            "017.txt": 0,
            "018.txt": 1,
            "019.txt": 0,
            "020.txt": 0,
            "021.txt": 0,
            "022.txt": 0,
            "023.txt": 0,
            "024.txt": 0,
            "025.txt": 0,
            "026.txt": 0,
            "027.txt": 1,
            "028.txt": 0,
            "029.txt": 0,
            "030.txt": 0,
            "031.txt": 0,
            "032.txt": 0,
            "033.txt": 0,
            "034.txt": 0,
        },
    }
    dir_list, allWordsArr, documentLengths, dfArr = load_documents(path)

    dfCounts = compute_df(dfArr)
    numDocs = len(dir_list)

    IDFDict = compute_idf(dfCounts, numDocs)

    avgdl = compute_avgdl(documentLengths)
    bm25IDF = compute_bm25_idf(dfCounts, numDocs)

    for query_id, query in queries.items():
        bm25Result = search_bm25(
            allWordsArr,
            documentLengths,
            bm25IDF,
            avgdl,
            query
        )

        rankingBM = list(bm25Result.keys())
        rankingsBM[query_id] = rankingBM

        print(query_id, query)
        print("RR:", reciprocal_rank(rankingBM, qrels[query_id]))

    for query_id, query in queries.items():
        tfidfResult = search_tfidf(allWordsArr, IDFDict, query)

        rankingTFIDF = list(tfidfResult.keys())
        rankingsTFIDF[query_id] = rankingTFIDF

        print(query_id, query)
        print("RR:", reciprocal_rank(rankingTFIDF, qrels[query_id]))

    print("BM25 MRR:", mean_reciprocal_rank(rankingsBM, qrels))
    print("TF-IDF MRR:", mean_reciprocal_rank(rankingsTFIDF, qrels))

    k_values = [1, 3, 5, 10]

    for k in k_values:
        print("BM25 P@", k)

        for query_id, ranking in rankingsBM.items():
            print(query_id, precision_at_k(ranking, qrels[query_id], k))

    for k in k_values:
        print("BM25 Recall@", k)

        for query_id, ranking in rankingsBM.items():
            print(query_id, recall_at_k(ranking, qrels[query_id], k))

    print()

    for k in k_values:
        print("BM25 Mean P@", k, ":", mean_metric_at_k(
            precision_at_k,
            rankingsBM,
            qrels,
            k
        ))

    for k in k_values:
        print("BM25 Mean Recall@", k, ":", mean_metric_at_k(
            recall_at_k,
            rankingsBM,
            qrels,
            k
        ))


    for k in k_values:
        print("TF-IDF Mean P@", k, ":", mean_metric_at_k(
            precision_at_k,
            rankingsTFIDF,
            qrels,
            k
        ))

    for k in k_values:
        print("TF-IDF Mean Recall@", k, ":", mean_metric_at_k(
            recall_at_k,
            rankingsTFIDF,
            qrels,
            k
        ))

    for k in k_values:
        print("TF-IDF Mean NDCG@", k, ":", mean_metric_at_k(
            ndcg_at_k,
            rankingsTFIDF,
            qrels,
            k
        ))

    for k in k_values:
        print("BM25 Mean NDCG@", k, ":", mean_metric_at_k(
            ndcg_at_k,
            rankingsBM,
            qrels,
            k
        ))

    # print_tfidf(allWordsArr, IDFDict)
    # print_tfidf_details(allWordsArr, documentLengths, IDFDict)

    # print(allWordsArr)

if __name__ == "__main__":
    main()