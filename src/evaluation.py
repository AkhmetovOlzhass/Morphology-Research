import math


def precision_at_k(ranking, qrels, k):
    allRank = ranking[:k]
    counter = 0
    for el in allRank:
        if(qrels[el] == 1):
            counter+=1
    return counter / k

def recall_at_k(ranking, qrels, k):
    allRank = ranking[:k]
    counter = 0
    allCounter = 0

    for el in allRank:
        if(qrels[el] == 1):
            counter+=1

    for el in ranking:
        if(qrels[el] == 1):
            allCounter+=1
    return counter / allCounter

def reciprocal_rank(ranking, qrels):
    found = False
    i = 0

    while i < len(ranking):
        if qrels[ranking[i]] == 1:
            found = True
            break

        i += 1

    if not found:
        return 0
    else:
        return 1 / (i + 1)

def mean_reciprocal_rank(rankings, qrels):
    total_rr = 0
    elements = len(rankings.keys())
    for el, arr in rankings.items():
        total_rr += reciprocal_rank(arr, qrels[el])
    return total_rr/elements

def mean_metric_at_k(metric_function, rankings, qrels, k):
    total = 0

    for query_id, ranking in rankings.items():
        total += metric_function(ranking, qrels[query_id], k)

    return total / len(rankings)


def dcg_at_k(ranking, qrels, k):
    i = 0
    DCGK = 0
    for i in range(k):
        reli = qrels[ranking[i]]
        DCGK += reli / (math.log2(i+2))
    return DCGK
            
def idcg_at_k(qrels, k):
    sorted_qrels = dict(sorted(qrels.items(), key=lambda item: item[1], reverse=True))
    ranking = list(sorted_qrels.keys())[:k]
    return dcg_at_k(ranking, qrels, k)
    
def ndcg_at_k(ranking, qrels, k):
    DCG = dcg_at_k(ranking, qrels, k)
    IDCG = idcg_at_k(qrels, k)

    if(IDCG == 0):
        return 0

    NDCG = DCG/IDCG
    return NDCG
