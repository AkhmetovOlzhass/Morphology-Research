# Baseline TF-IDF

## Purpose

Establish a simple information retrieval baseline before
introducing normalization, morphology, or BM25.

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

IDF(t) = log(N / DF(t))

## Score

Score(d,q) = sum(TF(t,d) × IDF(t))
for query terms occurring in the document.

## Known limitations

- raw TF favors repeated terms
- document length is not considered
- exact token matching is used
- morphological variants are treated as different terms
- case differences are treated as different terms
- no semantic similarity
- no phrase/proximity information