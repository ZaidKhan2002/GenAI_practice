# Day 17 — Similarity Search & Ranking

## 1. What is Similarity Search?

Similarity search finds documents whose vector representations are closest to a query vector.

Typical workflow:

1. Convert documents into embeddings.
2. Convert the user query into an embedding.
3. Calculate similarity between the query and document vectors.
4. Rank documents by similarity.
5. Retrieve the Top-K documents.

## 2. Similarity Metrics

### Cosine Similarity

Measures the angle/direction between two vectors.

\[
\text{cosine}(A,B)=\frac{A\cdot B}{\|A\|\|B\|}
\]

- +1: Same direction.
- 0: Orthogonal directions.
- -1: Opposite directions.

Higher cosine similarity generally indicates greater semantic similarity.

### Dot Product

\[
A\cdot B=\sum_i A_iB_i
\]

Measures both vector direction and magnitude.

Important: When both vectors are L2-normalised, dot product equals cosine similarity.

### Euclidean Distance

\[
d(A,B)=\sqrt{\sum_i(A_i-B_i)^2}
\]

Measures straight-line distance between vectors.

Lower distance means vectors are closer.

| Metric | Measures | Better result |
|---|---|---|
| Cosine similarity | Direction | Higher |
| Dot product | Direction and magnitude | Higher |
| Euclidean distance | Straight-line distance | Lower |

## 3. Top-K Retrieval

Top-K retrieval selects the K highest-ranked documents.

Examples:
- Top-1: One document.
- Top-5: Five documents.
- Top-10: Ten documents.

Choosing K involves a trade-off:
- Too small: Relevant information may be missed.
- Too large: Irrelevant context may enter the prompt and consume tokens.

## 4. Similarity Ranking vs Relevance Ranking

**Similarity ranking:** Measures how close query and document embeddings are in vector space.

**Relevance ranking:** Estimates how useful a document is for answering the actual query.

High semantic similarity does not guarantee that a document directly answers the question.

## 5. Reranking

A reranker evaluates the initially retrieved candidates more deeply.

Typical pipeline:

1. Retrieve Top-20 candidates using vector search.
2. Pass the query and candidates to a reranking model.
3. Reorder candidates by relevance.
4. Select Top-3 or Top-5 for the LLM context.

A cross-encoder reranker jointly processes the query and document, unlike a typical bi-encoder that independently generates embeddings.

## 6. Python Implementation

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Customers must repay loans according to the repayment schedule.",
    "Borrowers may request loan restructuring under specified conditions.",
    "Savings account interest is calculated quarterly.",
    "Late loan payments may attract penalty charges."
]

query = "What happens if a borrower misses a loan payment?"

document_embeddings = model.encode(
    documents,
    normalize_embeddings=True
)

query_embedding = model.encode(
    query,
    normalize_embeddings=True
)

# Normalised vectors: dot product = cosine similarity
scores = np.dot(document_embeddings, query_embedding)

ranked_indices = np.argsort(scores)[::-1]

top_k = 2
top_indices = ranked_indices[:top_k]

for idx in top_indices:
    print(documents[idx], scores[idx])
```

## 7. Interview Takeaways

- Cosine similarity compares vector direction.
- Raw dot product is affected by magnitude.
- L2-normalised vectors allow dot product to represent cosine similarity.
- Euclidean distance measures straight-line distance.
- Top-K controls the number of retrieved candidates.
- Similarity does not guarantee factual relevance.
- Reranking improves the ordering of retrieved candidates.
- Retrieval quality directly affects RAG answer quality.

## 8. Banking Example

Query: "What happens if a borrower misses a loan payment?"

A banking RAG system may retrieve policy documents about repayment schedules, late payment penalties and loan restructuring.

The retrieval stage finds candidates. A reranker can prioritise the policy clause that directly addresses late repayment penalties.

Use approved, access-controlled banking documents in production; synthetic documents are sufficient for learning and experimentation.