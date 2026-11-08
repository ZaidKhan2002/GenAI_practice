# Day 15 — Tokenisation and Embeddings

## 1. Why do LLMs need tokenisation?

Computers process numerical representations rather than raw human language. Tokenisation converts text into smaller units called **tokens**, which can subsequently be mapped to numerical token IDs.

Typical processing pipeline:

```text
Raw Text
   ↓
Tokenisation
   ↓
Tokens
   ↓
Token IDs
   ↓
Embedding Vectors
   ↓
Transformer Model
```

## 2. What is a token?

A token is a unit of text recognised by a tokenizer. Depending on the tokenizer, it can represent:

- A complete word.
- A subword.
- A punctuation mark.
- A special control token.

Example:

```text
Input: The customer failed to repay the loan.

Tokens:
['the', 'customer', 'failed', 'to',
 'repay', 'the', 'loan', '.']
```

A token is not necessarily equivalent to a complete word.

## 3. Types of tokenisation

### Word-level tokenisation

Splits text into complete words.

Example:

```text
"Banking is important"
→ ["Banking", "is", "important"]
```

Limitation: Requires a large vocabulary and struggles with rare or unseen words.

### Character-level tokenisation

Splits text into individual characters.

Example:

```text
"loan"
→ ["l", "o", "a", "n"]
```

Limitation: Produces long sequences and loses useful word-level structure.

### Subword tokenisation

Breaks uncommon words into smaller known units. It balances vocabulary size and the ability to represent rare words.

Common approaches:

- BPE (Byte Pair Encoding)
- WordPiece
- Unigram Language Model

## 4. WordPiece tokenisation

BERT uses WordPiece tokenisation.

WordPiece represents words using vocabulary entries that can be complete words or smaller subword units.

Example observed using `bert-base-uncased`:

```text
microfinance
→ ['micro', '##fin', '##ance']

unbankability
→ ['un', '##bank', '##ability']

electroencephalographically
→ ['electro', '##ence', '##pha',
   '##log', '##raphic', '##ally']
```

The `##` prefix indicates that the subword continues from a previous subword within the same word.

Important:
- WordPiece does not necessarily split words according to grammatical roots.
- It uses units available in its learned vocabulary.
- `[UNK]` is a special unknown token used when the tokenizer cannot represent an input using its vocabulary and tokenisation rules.

## 5. Token IDs

After tokenisation, every token is mapped to an integer ID from the tokenizer's vocabulary.

Example:

```text
[CLS]      → 101
the        → 1996
customer   → 8013
failed     → 3478
[SEP]      → 102
```

These IDs are vocabulary lookup indices.

**Important:** Token IDs do not directly represent semantic meaning. For example, ID 8013 is not inherently more meaningful than ID 1996.

## 6. Special tokens in BERT

BERT uses special tokens to structure its inputs.

| Token | Purpose |
|---|---|
| `[CLS]` | Special classification token placed at the beginning |
| `[SEP]` | Separates sequences and marks the end of a sequence |
| `[PAD]` | Pads shorter sequences to a common length |
| `[UNK]` | Represents an unknown token |
| `[MASK]` | Used in masked language modelling |

For a single sequence, BERT typically prepares:

```text
[CLS] The customer failed to repay the loan . [SEP]
```

Our experiment produced:

- 8 ordinary tokens.
- 2 special tokens.
- 10 total input token IDs.

## 7. Hugging Face tokenizer practical

File:

`15-Tokenization-and-Embeddings/practical/tokenization_demo.py`

Installation:

```bash
pip install transformers torch
```

Implementation:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "The customer failed to repay the loan."

# Convert text into tokens
tokens = tokenizer.tokenize(text)

# Generate model-ready tokenizer output
complete_output = tokenizer(text, return_tensors="pt")

# Extract token IDs
token_ids = complete_output["input_ids"]

# Convert IDs back into tokens
reconstructed_tokens = tokenizer.convert_ids_to_tokens(
    token_ids[0].tolist()
)

print("Tokens:", tokens)
print("Number of tokens:", len(tokens))
print("Token IDs:", token_ids)
print("Reconstructed tokens:", reconstructed_tokens)
print("Complete output:", complete_output)
print("Attention Mask:", complete_output["attention_mask"])
```

### Observed output

```text
Tokens:
['the', 'customer', 'failed', 'to',
 'repay', 'the', 'loan', '.']

Number of tokens: 8

Token IDs:
[101, 1996, 8013, 3478, 2000,
 24565, 1996, 5414, 1012, 102]

Reconstructed tokens:
['[CLS]', 'the', 'customer', 'failed', 'to',
 'repay', 'the', 'loan', '.', '[SEP]']
```

### Understanding tokenizer outputs

**`input_ids`**

Contains numerical vocabulary IDs corresponding to the input tokens, including applicable special tokens.

**`token_type_ids`**

Identifies the sequence segment to which each token belongs. In our single-sequence example, all values are zero.

**`attention_mask`**

Identifies valid tokens and padding positions:

- `1` = valid token.
- `0` = padding position.
