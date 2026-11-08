# Day 15 — Interview Questions and Answers

## Q1. Why do LLMs require tokenisation?

Tokenisation converts raw text into smaller units called tokens. These tokens are mapped to numerical IDs, which can then be converted into embeddings for processing by the Transformer.

## Q2. Why do modern LLMs use subword tokenisation instead of word-level tokenisation?

Word-level tokenisation requires a large vocabulary and struggles with rare or unseen words. Subword tokenisation breaks uncommon words into smaller known units, reducing vocabulary requirements while retaining the ability to represent many words.

## Q3. What is the difference between a token, token ID and embedding?

- Token: A unit of text.
- Token ID: An integer identifying that token in the tokenizer's vocabulary.
- Embedding: A dense numerical vector representing the token.

Token IDs are lookup indices, not semantic representations.

## Q4. What is WordPiece?

WordPiece is a subword tokenisation algorithm used by BERT. It represents words using vocabulary entries that may be complete words or subword units. The `##` prefix indicates that a subword continues from the preceding part of the same word.

## Q5. What is the purpose of `[CLS]` and `[SEP]` in BERT?

`[CLS]` is a special classification token added at the beginning of a sequence. `[SEP]` separates sequences and marks a sequence boundary.

## Q6. What is the difference between `input_ids` and `attention_mask`?

`input_ids` contains the numerical vocabulary IDs of tokens. `attention_mask` indicates valid input positions (`1`) and padding positions (`0`).

## Q7. Does an attention mask indicate which words are more important?

No. It identifies valid tokens versus padding positions. It does not measure semantic importance or guarantee how much attention a token receives.

## Q8. What are token type IDs?

Token type IDs identify the sequence segment to which each token belongs. In a typical BERT single-sequence input, all values are zero. In a paired-sequence input, they can distinguish the two segments.

## Q9. What is the difference between token embeddings and sentence embeddings?

Token embeddings represent individual tokens as vectors. Sentence embeddings represent an entire sentence or passage as one vector.

Token embeddings are used in Transformer input representations, while sentence embeddings are commonly used for semantic search and retrieval.

## Q10. Does a 1536-dimensional embedding always perform better than a 768-dimensional embedding?

No. Higher dimensionality provides more representational capacity but also increases storage and computational requirements. Actual performance depends on model architecture, training data, training objectives and suitability for the task.

## Q11. What happens when BERT encounters a rare word?

WordPiece attempts to split it into known subword units. If the tokenizer cannot represent it using its vocabulary and tokenisation rules, it may use the `[UNK]` token.

## Q12. Why might two sentences with different wording have similar sentence embeddings?

A suitable sentence-embedding model learns to represent semantic relationships. Therefore, sentences expressing similar meanings can have nearby vectors even when they use different words.

## Q13. Does WordPiece understand the contextual meaning of a word?

No. WordPiece performs tokenisation; it does not independently understand contextual meaning. Contextual representations are developed later by the Transformer through its attention and other layers.

## Q14. If a sentence produces 20 token IDs, will its sentence embedding have 20 dimensions?

No. Token count and embedding dimension are different concepts. A sentence may contain 20 tokens, while a sentence-embedding model produces a fixed-dimensional vector such as 384 or 768 dimensions.

## Q15. Why is tokenisation important in a RAG application?

Tokenisation helps models process queries and retrieved text. It also matters for chunk sizes and context-window limits. For semantic retrieval, a sentence-embedding model converts query and document chunks into vectors that can be compared.

## Interview scenario

**Question:** A banking policy document contains an uncommon financial term that does not exist as a complete token in BERT's vocabulary. Will BERT fail to process the document?

**Answer:** Not necessarily. WordPiece can break the term into smaller vocabulary entries. If it cannot represent the term using available subword units, it may produce `[UNK]`. This is why subword tokenisation is useful for handling rare terminology.