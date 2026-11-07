# Day 11 — Transformers and Attention Mechanism

## 1. Why Transformers?

RNNs process tokens sequentially and maintain a hidden state. They can struggle with long-range dependencies and parallel training.

LSTMs improve long-term dependency handling through gates but remain sequential.

Transformers use attention as their central mechanism, enabling parallel processing during training and better modelling of relationships between tokens.

## 2. Attention mechanism

Attention allows each token to incorporate information from other relevant tokens.

### Query, Key and Value

- **Query (Q):** What information is the token looking for?
- **Key (K):** What information does another token offer for matching?
- **Value (V):** What information is passed along and combined?

### Scaled dot-product attention

1. Calculate similarity scores using Q and K.
2. Divide scores by the square root of key dimension.
3. Apply Softmax to obtain attention weights.
4. Multiply weights by V to obtain contextual representations.

Formula:

\[
Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Scaling prevents excessively large dot-product scores, which could make Softmax too concentrated and gradients very small.

## 3. Self-attention vs cross-attention

- Self-attention: Q, K and V come from the same sequence.
- Cross-attention: Q comes from one sequence, while K and V come from another.

## 4. Multi-head attention

Multiple attention heads independently project the input into different Query, Key and Value representations.

Each head performs attention. Their outputs are concatenated and passed through a learned output projection.

\[
MultiHead(Q,K,V)=Concat(head_1,\ldots,head_h)W^O
\]

Benefits:
- Captures different relationships in parallel.
- Learns information from different representation subspaces.
- Combines the results into a unified representation.

Heads are not explicitly assigned fixed responsibilities; they learn patterns during training.

## 5. Positional encoding

Self-attention alone does not inherently encode token order.

Positional information is incorporated into token representations.

\[
Input = TokenEmbedding + PositionalEncoding
\]

Types:
- Sinusoidal positional encoding: calculated using sine and cosine functions.
- Learned positional embeddings: positional representations learned during training.
- Rotary Positional Embeddings (RoPE): incorporates positional information through rotations of Query and Key representations.

## 6. Transformer architecture

### Encoder block
- Multi-head self-attention.
- Residual connection and layer normalisation.
- Feed-forward network.
- Another residual connection and layer normalisation.

### Architecture variants

| Architecture | Example | Purpose |
|---|---|---|
| Encoder-only | BERT | Text understanding |
| Decoder-only | GPT | Autoregressive generation |
| Encoder-decoder | T5 | Input-to-output sequence transformation |

### Important components

- Residual connection: adds a block's input to its output, helping information and gradients flow.
- Layer normalisation: stabilises activations.
- Feed-forward network: applies learned transformations independently to each token representation.
- Causal masking: prevents tokens from attending to future positions.

## 7. GPT generation

GPT uses decoder-only Transformer blocks and predicts the next token based on previous tokens.

Process:
1. Tokenise input.
2. Create token embeddings and positional representations.
3. Process through Transformer blocks.
4. Produce logits for the next token.
5. Select or sample a token.
6. Feed the generated token into the next prediction step.

Training can process known sequence positions in parallel using causal masking. Autoregressive generation proceeds sequentially.

## 8. Practical implementation

File: `practical/scaled_dot_product_attention.py`

Implemented using NumPy:
- Stable Softmax.
- Query-Key matrix multiplication.
- Scaling by square root of key dimension.
- Attention weights.
- Weighted Value aggregation.

The implementation demonstrates one attention operation, not a complete trained Transformer.

## 9. Limitations and trade-offs

- Standard full self-attention has approximately O(n²) time and memory scaling with sequence length.
- Long context windows are computationally expensive.
- Autoregressive generation is sequential.
- Attention weights do not guarantee correct semantic understanding.

## 10. Key takeaway

Transformers use attention to model relationships between tokens. Multi-head attention learns multiple patterns, positional information preserves order, and decoder-only architectures such as GPT generate text autoregressively.