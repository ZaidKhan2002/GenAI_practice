# Day 17 — Quick Revision

## Key Formulas

**Cosine similarity**

\[
\frac{A\cdot B}{\|A\|\|B\|}
\]

**Dot product**

\[
A\cdot B=\sum_i A_iB_i
\]

**Euclidean distance**

\[
\sqrt{\sum_i(A_i-B_i)^2}
\]

## Quick Comparison

| Metric | Higher or lower? |
|---|---|
| Cosine similarity | Higher |
| Dot product | Higher |
| Euclidean distance | Lower |

## Five Things to Remember

1. Cosine similarity measures direction.
2. Raw dot product depends on magnitude and direction.
3. L2-normalised vectors make dot product equivalent to cosine similarity.
4. Top-K controls how many candidates are retrieved.
5. Similarity does not guarantee relevance.

## RAG Retrieval Pipeline

Query → Query Embedding → Vector Search → Top-K Candidates → Reranking → Relevant Context → LLM

## Day 17 Self-Assessment

- [x] Explain cosine similarity.
- [x] Differentiate cosine similarity and dot product.
- [x] Explain Euclidean distance.
- [x] Implement Top-K retrieval in Python.
- [x] Explain similarity vs relevance ranking.
- [x] Understand why reranking is useful.
