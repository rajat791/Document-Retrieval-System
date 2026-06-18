# Document Retrieval System

A Python implementation of a **Vector Space Model (VSM)** information retrieval system, evaluated against the CACM benchmark document collection. The system supports multiple term weighting schemes and configurable preprocessing pipelines, with full performance benchmarking across 20 configurations.

---

## Overview

The system ranks documents by relevance to a given query by computing cosine similarity between query and document vectors. It integrates with an `IR_engine.py` outer shell and implements the core retrieval logic in `my_retriever.py`, supporting three term weighting schemes: **Binary**, **TF**, and **TF-IDF**.

### Best Results (log-normalised TF-IDF + stemming + stop-list)

| Metric    | Score |
|-----------|-------|
| Precision | 0.28  |
| Recall    | 0.22  |
| F1        | 0.25  |

---

## Tech Stack

- **Language:** Python
- **Libraries:** `math` (standard library only — implemented from scratch)

---

## Architecture

The `Retrieve` class is initialised with an inverted index and a chosen term weighting scheme. Three values are precomputed at initialisation to avoid repeated runtime calculations:

- **`compute_number_of_documents`** — counts distinct document IDs across the index
- **`compute_idf`** — computes IDF for each term using `idf = log(N / df)`, where N is the total document count and df is the document frequency
- **`pre_compute_document_vector_lengths`** — precomputes `||d||` for every document, incorporating the chosen weighting scheme

---

## Pipeline

### 1. Query Processing
Each query is supplied as a token list and converted into a dictionary of term counts, e.g. `{'parallel': 2, 'computation': 1}`.

### 2. Vector Construction
Document vectors are constructed only over query terms (not the full vocabulary), keeping vectors sparse and efficient. For any term not present in a document, a value of 0 is appended.

### 3. Term Weighting

Three schemes are supported:

| Scheme            | Formula                        | Notes                                      |
|-------------------|--------------------------------|--------------------------------------------|
| Binary            | 1 if term present, else 0      | Ignores frequency entirely                 |
| TF                | `1 + log(tf)` if tf > 0        | Log-normalised to reduce dominance of frequent terms |
| TF-IDF            | `(1 + log(tf)) × idf`          | Balances local frequency with global rarity |

### 4. Cosine Similarity

Similarity between query vector **q** and document vector **d** is computed as:

```
cos(q, d) = Σ(qi × di) / (||q|| × ||d||)
```

`||d||` is precomputed at initialisation. Documents with no overlapping query terms receive a score of 0.

### 5. Ranking
All cosine scores are stored in a dictionary keyed by document ID, sorted in descending order, and the top 10 document IDs are returned.

---

## Performance Results

Evaluated across 4 preprocessing configurations:
- `—` : no preprocessing
- `-p` : stop-list only
- `-s` : stemming only
- `-s -p` : both stemming and stop-list

| Scheme           | — (none)        | -p (stop-list)  | -s (stemming)   | -s -p (both)    |
|------------------|-----------------|-----------------|-----------------|-----------------|
| Binary           | P:0.07 R:0.06 F1:0.06 | P:0.09 R:0.08 F1:0.08 | P:0.13 R:0.10 F1:0.11 | P:0.16 R:0.13 F1:0.15 |
| TF               | P:0.08 R:0.06 F1:0.07 | P:0.11 R:0.09 F1:0.10 | P:0.17 R:0.13 F1:0.15 | P:0.19 R:0.15 F1:0.17 |
| TF-IDF           | P:0.21 R:0.17 F1:0.18 | P:0.26 R:0.21 F1:0.23 | P:0.22 R:0.18 F1:0.19 | P:0.27 R:0.22 F1:0.24 |
| 1+log(TF)        | P:0.11 R:0.09 F1:0.10 | P:0.16 R:0.13 F1:0.14 | P:0.17 R:0.14 F1:0.16 | P:0.21 R:0.17 F1:0.19 |
| 1+log(TF)×IDF    | P:0.20 R:0.16 F1:0.18 | P:0.27 R:0.21 F1:0.24 | P:0.22 R:0.17 F1:0.19 | P:0.28 R:0.22 **F1:0.25** |

---

## Analysis

**Non-IDF schemes (Binary, TF, log-TF)** show an approximately linear increase in F1 as more preprocessing is applied — stop-listing and stemming progressively remove noise that masks relevant term matches.

- **Binary** scored lowest overall (F1: 0.06 → 0.15), as it ignores term frequency entirely. The 150% F1 improvement with full preprocessing shows how dependent it is on clean input.
- **Raw TF** improved on binary by incorporating frequency, but was hurt by common terms dominating scores. Full preprocessing yielded a 142% F1 increase.
- **Log-normalised TF** showed only marginal improvement over raw TF (+11% F1 with full preprocessing), likely because the CACM collection contains many short documents with low raw TF values — log normalisation has the most impact at higher TF values.

**IDF-based schemes** significantly outperformed non-IDF schemes by down-weighting common terms globally, making retrieval more discriminative from the start.

- **TF-IDF** was the best of the three main schemes across all configurations.
- **Log-normalised TF-IDF** achieved the highest overall F1 (0.25), though only a 4% improvement over standard TF-IDF with full preprocessing — diminishing returns at this scale.
- A minor drop in F1 was observed for IDF schemes when applying only a stop-list (no stemming). This may be because some removed stop-words contributed to term overlap and similarity scores in the IDF context.

---

## Running the System

```bash
# Run retrieval with TF-IDF weighting, stemming and stop-list
python IR_engine.py -w tfidf -s -p

# Evaluate results against the gold standard
python eval_ir.py cacm_gold_std.txt results.txt
```

---

## File Structure

```
├── my_retriever.py        # Core Retrieve class (VSM implementation)
├── IR_engine.py           # Outer retrieval shell (provided)
├── eval_ir.py             # Evaluation script (provided)
├── IR_data.pickle         # Preprocessed index and queries (provided)
├── cacm_gold_std.txt      # Relevance judgements benchmark (provided)
└── results.txt            # System output
```
