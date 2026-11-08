# Day 18 — Vector Databases

## 1. Why Vector Databases?

A vector database stores and retrieves numerical embeddings efficiently.

Compared with manually searching NumPy arrays, vector databases provide:
- Efficient similarity search.
- Vector indexing.
- Metadata filtering.
- Persistent storage.
- Support for large embedding collections.

## 2. Vector Record Structure

A typical vector record contains:

- ID: Unique identifier.
- Embedding: Numerical vector.
- Document: Original text or reference.
- Metadata: Additional information such as category, department and page number.

## 3. Important Concepts

### Vector Storage
Stores embeddings with associated identifiers, documents and metadata.

### Metadata Filtering
Restricts retrieval based on attributes such as document category or department.

Metadata filtering must not replace independent authorization checks.

### Indexing
Organises vectors to improve search efficiency.

### Approximate Nearest Neighbour (ANN)
Finds vectors close to a query without necessarily examining every vector.

Exact search provides true nearest neighbours; ANN trades some recall for faster retrieval.

## 4. FAISS vs ChromaDB

| Feature | FAISS | ChromaDB |
|---|---|---|
| Main purpose | Vector similarity search | Vector database |
| Indexing | Yes | Yes |
| Document storage | Managed separately | Built-in |
| Metadata filtering | Managed separately | Supported |
| Persistence | Save/load indexes | Persistent collections |

FAISS is a similarity-search library, whereas ChromaDB provides database-oriented storage and retrieval capabilities.

## 5. Vector Index Types

### Flat Index
- Exact nearest-neighbour search.
- Compares query against all stored vectors.
- Useful as a baseline.

### HNSW
- Graph-based ANN index.
- Navigates neighbouring vectors.
- Offers faster search at scale with memory and recall trade-offs.

FAISS examples:
- IndexFlatIP: Exact inner-product search.
- IndexFlatL2: Exact squared Euclidean-distance search.

## 6. FAISS Practical

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Customers must repay loans according to the repayment schedule.",
    "Borrowers may request loan restructuring under specified conditions.",
    "Savings account interest is calculated quarterly.",
    "Late loan payments may attract penalty charges."
]

embeddings = model.encode(
    documents,
    normalize_embeddings=True
).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

query = "What happens if a borrower misses a loan payment?"

query_embedding = model.encode(
    [query],
    normalize_embeddings=True
).astype("float32")

scores, indices = index.search(query_embedding, k=2)

for score, idx in zip(scores[0], indices[0]):
    print(documents[idx], score)
```

## 7. Practical Output

Top-1:
- Late loan payments may attract penalty charges.
- Similarity: 0.6537

Top-2:
- Customers must repay loans according to the repayment schedule.
- Similarity: 0.5256

The scores indicate vector similarity, not factual confidence.

## 8. Important Engineering Learning

Python module naming conflicts can occur when a local file has the same name as an imported package.

Example:
- Naming a file faiss.py can conflict with `import faiss`.
- Rename it to vector_search.py.
- Remove stale `__pycache__` files when necessary.

Use virtual environments to isolate project dependencies.

## 9. Interview Takeaways

- Vector databases store and retrieve embeddings.
- Indexing improves search efficiency.
- Exact search checks all vectors.
- ANN trades some recall for speed.
- FAISS is a vector search library.
- ChromaDB provides document, embedding and metadata storage.
- Normalised vectors make inner product equivalent to cosine similarity.
- Access control must be enforced separately from metadata filtering.