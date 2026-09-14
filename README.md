# Morphology-Aware Information Retrieval for Kazakh–Russian Code-Switched Queries

Исследование влияния морфологической вариативности и код-свитчинга на эффективность информационного поиска в казахско-русских запросах.

## Research question

Can morphology-aware query normalization improve information retrieval for Kazakh–Russian code-switched queries without fine-tuning the retrieval models?

## Hypothesis

Morphological normalization will reduce the retrieval gap between monolingual and code-switched queries, particularly for morphologically complex Kazakh queries.

## Motivation

Queries with the same information need can look very different:

```text
Қазақстанда өткен соңғы сайлау нәтижелері
Қазақстанда өткен соңғы election нәтижелері
Қазақстандағы сайлау нәтижесі қандай болды
```

For a person they are clearly related. For a retrieval system they may not match — especially when Kazakh agglutination changes word forms and a Russian component sits in another morphological system.

The project sits at the intersection of:

- low-resource NLP
- code-switching
- information retrieval
- morphology

## Controlled query conditions

For the same information need, compare:

| Condition | Description |
|-----------|-------------|
| Q1 | Kazakh query |
| Q2 | Russian translation |
| Q3 | Kazakh–Russian mixed query |
| Q4 | Mixed + morphological variants |

Goal: measure how language-sensitive and morphology-sensitive retrieval is, and whether query-side normalization closes the gap without training embeddings or LLMs.

## Method outline

1. **Corpus** — Kazakh / Russian–Kazakh news, KazQAD, and a small toy corpus for early experiments
2. **Queries** — controlled synthetic transformations (original → morphological variant → code-switched → Russian), plus a small human-validated set if available
3. **Baselines** — BM25, dense multilingual retrieval, hybrid, optional reranker
4. **Proposed method** — morphology-aware query normalization / expansion via Kazakh morphological analysis (lemma + features → candidate forms), not naive stemming
5. **Ablation** — BM25 / dense / hybrid with and without stemming, lemmatization, morphology expansion, and reranking

## Repository status

Early toy-corpus experiments:

| Experiment | Path | Notes |
|------------|------|-------|
| Baseline TF-IDF | [`experiments/baseline/`](experiments/baseline/) | raw TF × IDF, no length norm, no morphology |
| BM25 | [`experiments/bm25/`](experiments/bm25/) | length-aware ranking (`k1=1.5`, `b=0.75`) |

Code entry point: [`src/read_documents.py`](src/read_documents.py)

## Project layout

```text
data/           corpora (toy and later full datasets)
experiments/    experiment configs, logs, and ranked results
src/            retrieval pipeline
figures/        plots
literature/     notes on related work
notes/          working notes
paper/          paper drafts
results/        aggregated metrics
```

## License

See [LICENSE](LICENSE).
