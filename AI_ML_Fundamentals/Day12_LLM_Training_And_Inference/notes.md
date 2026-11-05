# Day 12 — LLM Training and Inference

## 1. What is an LLM?

A Large Language Model (LLM) is a model trained on large amounts of text to understand and generate language.

Most modern LLMs are based on Transformer architectures.

- Transformer: neural network architecture.
- LLM: a large-scale trained language model.
- Next-token prediction: a common training objective.

## 2. Next-token prediction

An LLM predicts the next token based on preceding tokens.

Example:

Input: `The customer opened`

Expected next token: `an`

The model calculates logits, converts them into probabilities using Softmax and learns to predict the actual next token.

### Self-supervised learning

The training targets are derived from the original text itself. Human-labelled examples are not required for every token.

## 3. Pretraining

Pretraining teaches general language patterns using large and diverse datasets.

Common sources include:
- Books
- Articles
- Websites
- Documentation
- Source code

A common objective is next-token prediction:

\[
L=-\sum_{t=1}^{T}\log P(x_t|x_{<t})
\]

The model adjusts its parameters to minimise prediction loss.

## 4. Fine-tuning

Fine-tuning further trains a pretrained model on targeted data.

Examples:
- Instruction following
- Domain-specific responses
- Structured JSON generation
- Classification and summarisation

Pretraining learns general patterns; fine-tuning adapts the model to particular tasks or behaviours.

## 5. SFT — Supervised Fine-Tuning

SFT trains the model using curated input-output examples.

Example:

Instruction: Explain fraud detection.

Expected response: Fraud detection identifies suspicious transactions using rules or machine learning models.

The model learns to generate responses resembling desired examples.

Instruction tuning commonly uses SFT.

## 6. RLHF — Reinforcement Learning from Human Feedback

Traditional RLHF pipeline:

1. Pretrained model
2. Supervised fine-tuning
3. Generate candidate responses
4. Human preference rankings
5. Train a reward model
6. Optimise the policy using reinforcement learning

Important terms:

- Reward model: scores responses based on learned preferences.
- Policy model: the LLM being optimised.
- PPO: Proximal Policy Optimization, a reinforcement-learning algorithm.
- KL penalty: helps constrain how far the optimised policy moves from a reference model.

### DPO — Direct Preference Optimization

DPO directly optimises a model using chosen and rejected response pairs.

Unlike the traditional reward-model-plus-PPO pipeline, DPO does not require that separate reward model and online RL optimisation loop.

## 7. RAG vs Fine-tuning

### Fine-tuning
- Updates model parameters.
- Useful for adapting behaviour and task performance.
- Requires additional training.

### RAG
- Retrieves relevant information from external sources.
- Supplies retrieved content to the model as context.
- Does not necessarily require model-weight updates.
- Useful for frequently changing information.

Example: Maintain current bank interest rates in an authorised database and retrieve the latest values when answering questions.

## 8. Inference

Inference is using a trained model to generate predictions without updating its weights during ordinary operation.

Typical inference pipeline:

1. User prompt
2. Tokenisation
3. Embeddings and positional information
4. Transformer layers
5. Logits
6. Softmax probabilities
7. Decoding
8. Generated token
9. Repeat until generation stops

### Autoregressive generation

Each newly generated token becomes part of the context for predicting the next token.

### KV Cache

Stores previously computed attention keys and values.

Benefits:
- Avoids unnecessary recomputation.
- Improves generation efficiency.
- Reduces latency in autoregressive inference.

Trade-off: KV Cache consumes memory and generally grows with sequence length.

## 9. Tokens and Context Window

A token is a unit of text processed by the model. It can represent a complete word, part of a word, punctuation or other text fragments.

A context window is the maximum token context a model can process in a request.

It may include:
- System and developer instructions
- User prompts
- Conversation history
- Retrieved documents
- Generated output, subject to model and serving limits

A larger context window does not guarantee persistent memory.

If the context limit is exceeded, an application may reject the request, truncate content or summarise older conversation history.

## 10. Decoding Strategies

### Softmax

Converts logits into probabilities:

\[
P_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

For numerical stability, subtract the maximum logit before exponentiation.

### Temperature

\[
P_i=\operatorname{softmax}(z_i/T)
\]

- Low temperature: sharper distribution, more predictable sampling.
- Temperature 1: original Softmax distribution.
- High temperature: flatter distribution, more varied sampling.

Temperature does not increase model knowledge or guarantee correctness.

### Greedy decoding

Always selects the token with the highest probability.

For positive temperature, changing temperature alone does not change the ranking of logits, so greedy selection remains unchanged.

### Top-K

Retains a fixed number of highest-probability candidates and renormalises their probabilities.

### Top-P (Nucleus Sampling)

Retains the smallest set of highest-probability tokens whose cumulative probability reaches or exceeds the P threshold.

The token crossing the threshold must be included.

## 11. Comparison

| Concept | Purpose |
|---|---|
| Pretraining | Learn general language patterns |
| SFT | Learn desired instruction-response behaviour |
| RLHF | Optimise behaviour using human preferences |
| DPO | Direct preference-based optimisation |
| RAG | Retrieve external information |
| Inference | Generate predictions using trained weights |
| KV Cache | Avoid repeated attention computation |
| Temperature | Control probability distribution sharpness |
| Top-K | Fixed candidate selection |
| Top-P | Cumulative probability-based candidate selection |

## 12. Key interview takeaways

- Pretraining is commonly self-supervised.
- Fine-tuning adapts an already-trained model.
- SFT teaches through examples.
- RLHF uses human preferences.
- DPO directly optimises preference pairs.
- RAG provides external context without necessarily updating weights.
- Inference ordinarily does not update model weights.
- Context window is not persistent memory.
- KV Cache improves autoregressive inference efficiency.
- Temperature, Top-K and Top-P are decoding strategies.