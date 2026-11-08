# Day 17 — Interview Questions

## Q1. What is similarity search?

Similarity search retrieves documents whose vector embeddings are closest to a query embedding according to a selected similarity metric.

## Q2. What is the difference between cosine similarity and dot product?

Cosine similarity measures vector direction, while dot product is affected by both direction and magnitude. For L2-normalised vectors, they are equivalent.

## Q3. Why is cosine similarity commonly used in semantic search?

It focuses on vector direction rather than magnitude, making it useful when the orientation of embeddings represents semantic similarity.

## Q4. What is Euclidean distance?

It measures straight-line distance between two vectors. Lower distance indicates closer vectors.

## Q5. What is Top-K retrieval?

It retrieves the K highest-ranked candidates according to a search metric.

## Q6. Can a similarity score of 0.95 guarantee that two documents mean exactly the same thing?

No. It indicates high vector similarity, not factual equivalence or guaranteed relevance.

## Q7. Why can retrieving too many documents hurt RAG performance?

It can introduce irrelevant information, consume context-window tokens, increase latency and potentially distract the generation model.

## Q8. What happens if query and document embeddings use incompatible embedding models?

Their vector spaces or dimensions may be incompatible. Similarity calculations may be invalid or unreliable.

## Q9. What is the difference between similarity ranking and relevance ranking?

Similarity ranking measures geometric closeness between embeddings. Relevance ranking estimates how useful a document is for answering the user's actual question.

## Q10. What is reranking?

Reranking evaluates initially retrieved candidates and reorders them according to their relevance to the query.

## Q11. What is a cross-encoder reranker?

A cross-encoder processes the query and document together to produce a relevance score. It can capture interactions between them more directly than independent embedding generation.

## Q12. Why might a RAG pipeline use Top-20 retrieval followed by Top-5 reranking?

Initial vector search efficiently narrows the candidate set. A more computationally expensive reranker then identifies the most relevant documents.

## Q13. Does a higher cosine similarity always mean a better answer?

No. Semantic similarity and answer relevance are different. A document can discuss the same subject without containing the answer.

## Q14. How would you improve poor retrieval quality in a banking RAG system?

Investigate:
- Embedding model suitability.
- Document parsing and chunking.
- Query formulation.
- Metadata filtering.
- Hybrid retrieval.
- Top-K selection.
- Reranking.
- Retrieval evaluation using representative queries.

## Q15. Explain the retrieval flow in a banking RAG application.

1. Embed the user's query.
2. Search the vector database.
3. Retrieve the top candidate chunks.
4. Apply metadata and access-control filters.
5. Rerank candidates.
6. Select relevant context.
7. Pass context to the LLM.
8. Generate a grounded answer with appropriate citations.