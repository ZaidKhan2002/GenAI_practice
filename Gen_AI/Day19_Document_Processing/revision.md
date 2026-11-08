# Day 19 — Revision

## Quick Revision

| Concept | Remember |
|---|---|
| Ingestion | Load, extract, clean, chunk, attach metadata, embed and store |
| OCR | Extracts text from scanned images |
| Cleaning | Removes unwanted formatting while preserving meaning |
| Chunking | Splits documents into smaller retrieval units |
| Chunk size | Controls the amount of text per chunk |
| Overlap | Preserves context between neighbouring chunks |
| Fixed chunking | Uses predefined sizes |
| Recursive chunking | Splits using natural boundaries |
| Semantic chunking | Uses meaning and topic changes |
| Metadata | Supports filtering, citations and traceability |

## Important Formulas

**Chunk step size:**

`step = chunk_size - overlap`

**Valid parameters:**

- `chunk_size > 0`
- `overlap >= 0`
- `overlap < chunk_size`

## Common Interview Traps

1. Chunking is not primarily about speeding up embedding generation; it improves retrieval granularity.
2. Excessive overlap increases duplicate retrieval and storage costs.
3. A scanned PDF may not contain extractable text.
4. Metadata filtering alone does not guarantee access security.
5. Larger chunks are not automatically better.
6. Tables must retain their structural relationships during extraction.

## Banking RAG Example

Question: What penalty applies to a late loan payment?

A RAG application should:
1. Retrieve relevant loan-policy chunks.
2. Check that the user is authorised to access the source.
3. Pass relevant content to the LLM.
4. Generate an answer grounded in retrieved information.
5. Include the policy document and page reference.

## Practical Revision Checklist

- [ ] Load a synthetic document.
- [ ] Clean unnecessary whitespace.
- [ ] Implement fixed-size chunking.
- [ ] Configure chunk overlap.
- [ ] Validate chunk-size parameters.
- [ ] Attach metadata to each chunk.
- [ ] Inspect output for missing spaces or broken text.
