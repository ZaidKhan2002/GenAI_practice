# Day 16 — Revision

## Concepts checklist

- [x] Definition and purpose of embedding models.
- [x] Text-to-vector processing pipeline.
- [x] Token embeddings and contextual representations.
- [x] Transformer encoder and pooling.
- [x] Mean pooling, CLS pooling and max pooling.
- [x] Embedding model vs generative LLM.
- [x] Document embeddings vs query embeddings.
- [x] Embedding dimensions.
- [x] Common embedding models.
- [x] Cosine similarity fundamentals.

## Practical checklist

- [x] Install Sentence Transformers.
- [x] Load `all-MiniLM-L6-v2`.
- [x] Generate embeddings for three sentences.
- [x] Inspect embedding dimensions.
- [x] Print an individual sentence embedding.
- [x] Calculate cosine similarity between related sentences.
- [x] Compare related and unrelated sentences.

## Practical results

| Test | Result |
|---|---:|
| Number of sentences | 3 |
| Embedding dimension | 384 |
| Total vector values | 1,152 |
| Similarity: loan vs missed payment | 0.64909804 |
| Similarity: loan vs weather | -0.048483685 |

## Interview revision points

1. Embedding model converts text into vectors.
2. Pooling converts token representations into a sentence representation.
3. Generative LLMs and embedding models have different responsibilities.
4. Query and document embeddings must be compatible.
5. Higher embedding dimensions do not guarantee better quality.
6. Cosine similarity measures vector direction similarity.
7. Similarity scores are not probabilities.
8. Retrieval quality must be evaluated using representative queries.

## Day 16 completion

- [x] Theory covered.
- [x] Practical implementation completed.
- [x] Embedding dimensions understood.
- [x] Similarity comparisons completed.
- [x] Interview concepts reviewed.
