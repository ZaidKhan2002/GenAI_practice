# Day 12 — Quick Revision Sheet

## Training

| Term | One-line meaning |
|---|---|
| LLM | Large-scale language model |
| Next-token prediction | Predict the next token from preceding context |
| Self-supervised | Targets derived from original data |
| Pretraining | Learn general language patterns |
| Fine-tuning | Adapt an existing model |
| SFT | Learn from desired input-output examples |
| RLHF | Improve responses using human preferences |
| Reward model | Scores responses based on preferences |
| PPO | Reinforcement-learning optimisation algorithm |
| DPO | Direct preference optimisation |

## RAG

- Retrieves relevant external information.
- Supplies retrieved content as context.
- Does not necessarily update model weights.
- Useful for frequently changing information.

## Inference

**Pipeline:**

Prompt → Tokenisation → Embeddings → Transformer → Logits → Softmax → Decoding → Next token.

- Inference ordinarily does not update weights.
- Autoregressive generation predicts one token at a time.
- KV Cache reuses previous keys and values.
- Context window is not permanent memory.

## Context window

May contain:
- Instructions
- User prompt
- Conversation history
- Retrieved documents
- Generated output, subject to the serving limit

Exceeding the limit can cause rejection, truncation or application-level context management.

## Decoding

| Technique | Remember |
|---|---|
| Softmax | Logits to probabilities |
| Temperature | Controls distribution sharpness |
| Greedy | Highest-probability token |
| Top-K | Fixed number of candidates |
| Top-P | Cumulative probability threshold |

### Formulas

Softmax:

\[
P_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

Temperature:

\[
P_i=\operatorname{softmax}(z_i/T)
\]

Pretraining objective:

\[
L=-\sum_{t=1}^{T}\log P(x_t|x_{<t})
\]

## Common interview traps

1. A pretrained model is not automatically a helpful instruction-following assistant.
2. Fine-tuning is not the same as RAG.
3. RAG does not inherently change model weights.
4. Ordinary inference does not retrain the model.
5. A token is not necessarily a complete word.
6. Context window is not persistent memory.
7. Temperature does not guarantee factual correctness.
8. Low temperature is not a security mechanism.
9. Top-P must include the token that crosses the threshold.
10. Greedy decoding is unaffected by positive temperature changes alone, assuming no other decoding constraints.



Role: Act as an application incident analyst

Perform the following tasks:
1. Identify the symptoms and relevant evidence.
2. Suggest possible causes without claiming certainty.
3. Distinguish confirmed facts from hypotheses.
4. Recommend immediate investigation steps.

Return the structured json in the below format.
{
  "incident_id": "string",
  "severity": "LOW | MEDIUM | HIGH | CRITICAL",
  "symptoms": ["string"],
  "evidence": ["string"],
  "possible_causes": ["string"],
  "confirmed_root_cause": "string or null",
  "recommended_actions": ["string"],
  "requires_human_review": true
}

