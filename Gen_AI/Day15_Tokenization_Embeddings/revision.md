# Day 15 — Revision Checklist

## Core concepts

- [ ] Explain why tokenisation is required.
- [ ] Differentiate word-level, character-level and subword tokenisation.
- [ ] Explain BPE, WordPiece and Unigram at a high level.
- [ ] Explain the meaning of the `##` prefix.
- [ ] Explain token vs token ID vs embedding.
- [ ] Explain BERT special tokens: `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`, `[MASK]`.
- [ ] Understand `input_ids`, `token_type_ids` and `attention_mask`.
- [ ] Differentiate `tokenizer.tokenize()` and `tokenizer()`.
- [ ] Explain token embeddings vs sentence embeddings.
- [ ] Explain embedding dimensions and why larger is not automatically better.

## Practical revision

- [x] Load BERT tokenizer using Hugging Face.
- [x] Convert a sentence into tokens.
- [x] Generate token IDs.
- [x] Reconstruct tokens from IDs.
- [x] Inspect complete tokenizer output.
- [x] Inspect attention mask.
- [x] Test uncommon words using WordPiece.
- [ ] Independently explain each output without referring to notes.
- [ ] Complete the embedding-generation practical in Day 15's next learning section.

## Important interview reminders

1. Token IDs are vocabulary indices, not semantic values.
2. Token count is not the same as embedding dimension.
3. Attention mask indicates valid input positions, not token importance.
4. WordPiece does not understand contextual meaning.
5. Sentence embeddings represent whole text segments.
6. Higher embedding dimensions do not guarantee better retrieval quality.

