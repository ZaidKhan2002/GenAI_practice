# Day 11 — Interview Questions

## Fundamentals

**1. Why were Transformers introduced?**

RNNs process tokens sequentially and can struggle with long-range dependencies. Transformers use attention to model token relationships and allow parallel processing during training.

**2. What are Query, Key and Value?**

Query represents what a token is looking for, Key represents information used for matching, and Value contains the information that gets combined.

**3. Explain scaled dot-product attention.**

Calculate Q multiplied by K transpose, scale by the square root of key dimension, apply Softmax, and multiply the resulting weights by V.

**4. Why divide by sqrt(d_k)?**

To control the magnitude of dot-product scores and avoid overly concentrated Softmax distributions with very small gradients.

**5. What is self-attention?**

An attention mechanism where Query, Key and Value originate from the same input sequence.

**6. What is cross-attention?**

An attention mechanism where Query comes from one sequence and Key and Value come from another.

## Multi-head attention

**7. Why use multiple attention heads?**

They allow the model to capture different relationships from multiple representation subspaces in parallel.

**8. What happens after individual attention heads finish?**

Their outputs are concatenated and passed through a learned output projection.

**9. Are attention heads assigned fixed responsibilities?**

No. They learn attention patterns during training.

## Positional encoding

**10. Why is positional encoding required?**

Self-attention alone has no inherent awareness of token order. Positional information helps distinguish different token positions.

**11. Difference between sinusoidal and learned positional encoding?**

Sinusoidal values are generated using fixed mathematical functions. Learned positional embeddings are updated during training.

**12. What is RoPE?**

Rotary Positional Embeddings incorporate positional information by rotating Query and Key representations according to token positions.

## Transformer architecture

**13. Difference between encoder-only, decoder-only and encoder-decoder?**

- Encoder-only: primarily input understanding.
- Decoder-only: autoregressive text generation.
- Encoder-decoder: transforms an input sequence into an output sequence.

**14. What is causal masking?**

It prevents a token from attending to future positions, enabling autoregressive next-token prediction without future-token leakage.

**15. What is a residual connection?**

It adds a block's input to its output, helping preserve information and improve gradient flow.

**16. What is a feed-forward network in a Transformer?**

It applies learned transformations independently to each token representation after attention.

**17. Why can Transformer training be parallelised but generation is sequential?**

During training, the complete target sequence is available and causal masking controls which positions can attend to one another. During generation, each new token depends on previously generated tokens.

## Advanced questions

**18. What is the computational complexity of standard self-attention?**

Approximately O(n²) in sequence length for attention scores and their associated memory.

**19. What is autoregressive generation?**

Generating one token at a time, conditioning each prediction on the preceding tokens.

**20. Does attention guarantee that a model understands a sentence?**

No. Attention computes weighted information aggregation. It does not guarantee factual correctness or human-like understanding.