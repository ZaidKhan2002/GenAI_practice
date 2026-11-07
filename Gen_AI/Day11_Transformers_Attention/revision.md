# Day 11 — Quick Revision

## Core concepts

| Concept | Remember |
|---|---|
| RNN | Sequential token processing |
| LSTM | RNN with gates for longer dependencies |
| Transformer | Attention-based architecture |
| Query | What am I looking for? |
| Key | What am I matching against? |
| Value | What information should I combine? |
| Softmax | Converts scores into normalised weights |
| Self-attention | Q, K, V from same sequence |
| Cross-attention | Q from one sequence; K, V from another |
| Multi-head attention | Multiple attention calculations in parallel |
| Positional encoding | Adds token-order information |
| RoPE | Rotary positional information |
| Causal mask | Blocks future-token attention |
| Residual connection | Adds input to block output |
| Layer norm | Normalises activations |
| FFN | Token-wise learned transformation |

## Must-remember formula

\[
Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

## Multi-head attention

\[
MultiHead(Q,K,V)=Concat(head_1,\ldots,head_h)W^O
\]

## Architecture comparison

- BERT: encoder-only.
- GPT: decoder-only.
- T5: encoder-decoder.

## GPT generation flow

Input → Tokenisation → Embeddings + Position → Transformer blocks → Logits → Next token → Repeat.

## Important interview distinctions

1. Training can parallelise positions; autoregressive generation is sequential.
2. Attention weights are not the same as Value vectors.
3. Self-attention does not inherently encode sequence order.
4. Causal masking prevents future-token leakage.
5. Multi-head outputs are concatenated and projected.
6. Attention is not a guarantee of factual correctness.

## Practical

`practical/scaled_dot_product_attention.py`

Implemented:
- Stable Softmax.
- Q × K transpose.
- Scaling.
- Attention weights.
- Weighted Value aggregation.

## Day 11 status

- [x] RNN and LSTM fundamentals
- [x] Attention and Q/K/V
- [x] Self-attention and cross-attention
- [x] Multi-head attention
- [x] Positional encoding
- [x] Transformer architecture
- [x] Causal masking
- [x] NumPy attention implementation
- [x] Final interview assessment
