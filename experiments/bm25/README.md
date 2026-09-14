# BM25

## Purpose

Compare a length-aware ranking function against the
raw TF-IDF baseline on the same toy corpus and queries.

## Corpus

Toy corpus of manually created Kazakhstan news documents.

## Tokenization

Whitespace tokenization after basic punctuation removal.

## TF

Raw term frequency:

TF(t,d) = number of occurrences of term t in document d.

## DF

Number of documents containing the term.

## IDF

IDF(t) = log((N - DF(t) + 0.5) / (DF(t) + 0.5))

## Score

Score(d,q) = sum over query terms t in d:

IDF(t) × (TF(t,d) × (k1 + 1)) / (TF(t,d) + k1 × (1 - b + b × (|d| / avgdl)))

## Parameters

- k1 = 1.5
- b = 0.75

## Known limitations

- exact token matching is used
- morphological variants are treated as different terms
- case differences are treated as different terms
- no semantic similarity
- no phrase/proximity information
- BM25 saturates repeated terms, but does not add morphology
