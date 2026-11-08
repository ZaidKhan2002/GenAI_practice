# Day 18 — Interview Questions

## Q1. Why do we need vector databases?
They provide efficient vector retrieval, indexing, metadata filtering and persistence.

## Q2. What is ANN?
Approximate Nearest Neighbour search retrieves close vectors without necessarily checking every stored vector.

## Q3. Exact search vs ANN?
Exact search guarantees true nearest neighbours. ANN is generally faster but may miss some true neighbours.

## Q4. Why does IndexFlatIP work with cosine similarity?
When vectors are L2-normalised, their inner product equals cosine similarity.

## Q5. FAISS vs ChromaDB?
FAISS focuses on high-performance vector similarity search. ChromaDB offers a database-oriented experience with document and metadata storage.

## Q6. What is HNSW?
A graph-based approximate nearest-neighbour indexing method.

## Q7. What is metadata filtering?
Filtering search results using attributes associated with documents, such as category or department.

## Q8. Does metadata filtering guarantee access control?
No. Backend authorization must independently restrict which documents the user can retrieve.

## Q9. Why is IndexFlatIP called an exact index?
It performs inner-product comparisons against all stored vectors and returns the exact highest-scoring results.

## Q10. What happens if we store embeddings from incompatible models in the same search space?
Similarity comparisons may be invalid or unreliable because the vectors may belong to incompatible embedding spaces.