# Day 12 — LLM Interview Questions

## Fundamentals

**1. What is an LLM?**

An LLM is a large-scale language model trained on text data to understand and generate language. Modern LLMs commonly use Transformer architectures.

**2. What is next-token prediction?**

It is the process of predicting the next token based on preceding tokens. It is a common objective used to train autoregressive LLMs.

**3. Why is next-token prediction self-supervised?**

Because the original text provides the target token automatically. Human-labelled data is not required for each training example.

**4. What is the difference between pretraining and fine-tuning?**

Pretraining learns general language patterns from large datasets. Fine-tuning further trains the pretrained model on targeted examples to adapt its behaviour or task performance.

## Alignment and Fine-tuning

**5. What is SFT?**

Supervised Fine-Tuning trains a pretrained model using curated instruction-response examples so it learns to follow instructions and produce desired outputs.

**6. What is instruction tuning?**

It is commonly SFT performed using instruction-response datasets.

**7. What is RLHF?**

Reinforcement Learning from Human Feedback uses human preference information to improve model responses.

**8. What is the role of a reward model?**

It learns to score responses based on preference comparisons and provides a reward signal for policy optimisation.

**9. What is PPO?**

Proximal Policy Optimization is a reinforcement-learning algorithm used in traditional RLHF pipelines to optimise the policy model.

**10. What is DPO?**

Direct Preference Optimization directly optimises a model using chosen and rejected response pairs without requiring the traditional separate reward-model-plus-PPO pipeline.

**11. Difference between SFT and RLHF?**

SFT learns from desired examples. RLHF optimises responses using human preference feedback.

## RAG and Model Knowledge

**12. When would you choose RAG over fine-tuning?**

When an application needs frequently changing external information, such as current interest rates or updated policy documents. RAG retrieves the latest information instead of requiring model retraining.

**13. Does RAG update model weights?**

Not necessarily. In a typical RAG system, retrieved documents are supplied as context during inference.

**14. Can fine-tuning replace RAG completely?**

No. Fine-tuning and RAG solve different problems. Fine-tuning adapts model behaviour, while RAG supplies relevant external information.

## Inference and Architecture

**15. What is inference?**

Inference is using an already-trained model to generate predictions or responses without updating its weights during ordinary operation.

**16. Explain autoregressive generation.**

The model predicts one token at a time. Each generated token becomes part of the context for predicting the next token.

**17. What is KV Cache?**

KV Cache stores previously calculated attention keys and values so the model can reuse them during autoregressive generation.

**18. What is a context window?**

It is the maximum token context that a model can process in a request. It can include instructions, prompts, conversation history, retrieved documents and generated output, depending on the model and serving configuration.

**19. Is context window the same as memory?**

No. Context is information available to the model for a request. Persistent memory requires an additional storage or application mechanism.

**20. What happens when the context limit is exceeded?**

The application may reject the request, truncate content, or summarise and remove older context.

## Decoding Strategies

**21. What are logits?**

Logits are raw scores produced by the model for possible next tokens. Softmax converts them into probabilities.

**22. What is temperature?**

Temperature controls the sharpness of the probability distribution. Lower values make sampling more predictable; higher values generally make it more varied.

**23. Does increasing temperature change greedy decoding?**

Not necessarily. For positive temperatures, scaling logits preserves their ranking, so greedy decoding selects the same highest-ranked token.

**24. What is Top-K sampling?**

It retains a fixed number of highest-probability candidate tokens and renormalises their probabilities.

**25. What is Top-P sampling?**

It retains the smallest set of highest-probability candidates whose cumulative probability reaches or exceeds the specified threshold.

**26. Difference between Top-K and Top-P?**

Top-K uses a fixed candidate count. Top-P uses a cumulative probability threshold, so the number of retained candidates can vary.

**27. Are temperature, Top-K and Top-P training techniques?**

No. They are inference-time decoding strategies.

## Practical and Design Questions

**28. Why use low temperature for banking classification?**

It generally produces more predictable sampling. However, reliability and security also require validation, authorisation, appropriate model evaluation and guardrails.

**29. Can an LLM hallucinate even with a large context window?**

Yes. A large context window does not guarantee factual accuracy or correct use of the supplied context.

**30. How would you reduce repeated computation during token generation?**

Use KV Cache to reuse previously computed attention keys and values.

## Important distinctions to revise

- Pretraining vs fine-tuning
- SFT vs RLHF
- PPO vs DPO
- RAG vs fine-tuning
- Training vs inference
- Context window vs persistent memory
- Logits vs probabilities
- Greedy vs sampling
- Top-K vs Top-P