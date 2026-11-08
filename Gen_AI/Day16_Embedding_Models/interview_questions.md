# Day 16 — Interview Questions and Answers

## Q1. What is an embedding model?

An embedding model converts text into dense numerical vectors that capture useful semantic information. These vectors are used in semantic search, similarity comparison and RAG retrieval.

## Q2. How does an embedding model generate sentence embeddings?

The text is tokenised and converted into token IDs. An embedding layer creates token vectors, which are processed by a Transformer encoder. Pooling then combines the token representations into a single sentence or passage vector.

## Q3. What is the role of pooling?

Pooling combines token-level representations into one fixed-dimensional vector representing the complete sentence or passage.

Common approaches include mean pooling, CLS pooling and max pooling.

## Q4. What is the difference between an embedding model and a generative LLM?

An embedding model converts text into numerical vectors for semantic representation and retrieval. A generative LLM predicts and generates text.

In RAG, the embedding model supports retrieval, while the LLM generates the final response using retrieved context.

## Q5. What is the difference between token embeddings and sentence embeddings?

Token embeddings represent individual tokens as vectors. Sentence embeddings represent an entire sentence or passage as one vector.

Sentence embeddings are commonly used in semantic search and RAG.

## Q6. Why do two sentences with different wording sometimes have similar embeddings?

Embedding models are trained to capture useful semantic relationships. Consequently, sentences with similar meanings can have nearby representations even when they use different vocabulary.

## Q7. What is embedding dimension?

Embedding dimension is the number of numerical values in a vector.

For example, a 384-dimensional embedding contains 384 numerical values.

Higher dimensions do not automatically guarantee better retrieval quality.

## Q8. What is cosine similarity?

Cosine similarity measures the directional similarity between two vectors using the cosine of the angle between them.

- +1: Same direction.
- 0: Orthogonal.
- -1: Opposite directions.

It is commonly used to compare query and document embeddings.

## Q9. Can cosine similarity be negative?

Yes. A negative score indicates that the vectors point in different directions. It does not necessarily mean the corresponding sentences have opposite meanings.

## Q10. Can we use different embedding models for queries and documents?

Generally, query and document embeddings must belong to compatible vector spaces and have compatible dimensions.

Some retrieval models explicitly support separate query and document encoders or prefixes. The model's documented conventions must be followed.

## Q11. Why not use a generative LLM directly for semantic search?

Generative LLMs are designed primarily for generating responses. Embedding models provide vector representations that can be compared and indexed efficiently for semantic retrieval.

Separating retrieval and generation also makes the RAG pipeline easier to evaluate and optimise.

## Q12. How would you select an embedding model for a banking RAG application?

I would evaluate:

1. Retrieval quality on representative banking queries and documents.
2. Support for domain terminology and required languages.
3. Embedding dimensions and storage requirements.
4. Latency and throughput.
5. Licensing and deployment requirements.
6. Data privacy and security constraints.

I would compare candidate models using a fixed retrieval evaluation dataset rather than selecting one based only on its advertised embedding dimension.

## Q13. What is the difference between embedding generation and similarity search?

Embedding generation converts text into vectors. Similarity search compares a query vector against stored document vectors and ranks the most relevant candidates.

## Q14. What is the role of embeddings in RAG?

Embeddings represent document chunks and user queries as vectors. The retrieval system compares these vectors to identify relevant document chunks. The retrieved context is then passed to the LLM for answer generation.

## Practical interview scenario

**Question:** A banking application has two documents with different wording but nearly identical policy meaning. How can semantic search retrieve the relevant document?

**Answer:** The application generates embeddings for the document chunks and the user query. A similarity-search mechanism compares their vectors and ranks the closest candidates. Since the embedding model captures semantic relationships, it can retrieve relevant documents even when the exact keywords differ.

The retrieval result must still be evaluated for correctness and relevance.