# Day 1 — Quick Revision Sheet

## 1. Concept Hierarchy

```text
Artificial Intelligence (AI)
          |
    Machine Learning (ML)
          |
      Deep Learning (DL)
          |
  Many modern GenAI models
```

Note: This is a simplified conceptual hierarchy. Not every generative technique necessarily belongs to this exact chain.

## 2. One-Line Definitions

| Term | Quick definition |
|---|---|
| AI | Building systems capable of intelligent behaviour |
| ML | Learning patterns from data |
| DL | ML using multi-layered neural networks |
| GenAI | Generating new content |
| LLM | Model designed for language understanding and generation |
| Dataset | Collection of examples |
| Feature | Input characteristic |
| Label | Expected target value |
| Algorithm | Method used to learn |
| Model | Learned representation |
| Training | Learning from data |
| Inference | Generating output using a trained model |

## 3. Types of Machine Learning

| Type | Data | Example |
|---|---|---|
| Supervised | Labelled | Spam classification |
| Unsupervised | Unlabelled | Customer segmentation |
| Reinforcement | Rewards and penalties | Game-playing agent |

## 4. Neural Network Formula

```text
z = w1*x1 + w2*x2 + b

a = f(z)
```

- x: Input feature
- w: Weight
- b: Bias
- f: Activation function
- a: Output

### Neural Network Training

```text
Input
  |
  v
Forward Propagation
  |
  v
Prediction
  |
  v
Loss Calculation
  |
  v
Backpropagation
  |
  v
Gradient Descent
  |
  v
Update Weights and Biases
  |
  +---- Repeat
```

Remember:
- Forward propagation calculates predictions.
- Loss measures prediction error.
- Backpropagation calculates gradients.
- Gradient descent updates parameters.

## 5. Rule-Based vs ML

| Rule-Based | ML |
|---|---|
| Explicit conditions | Learns from examples |
| Manually defined logic | Learned model parameters |
| Predictable rules | Statistical predictions |
| Manual rule updates | Retraining and monitoring |

**Banking example:**

Rule: Flag transactions above ₹100,000 with at least three failed logins.

ML: Learn suspicious transaction patterns from historical transaction data.

Hybrid: Combine policy rules with ML risk scores.

## 6. Important GenAI Distinctions

- An LLM is one category of Generative AI.
- Not all GenAI systems are LLMs.
- ML can exist without GenAI.
- Rule-based automation can exist without ML.
- Not every component in a GenAI application is AI.

## 7. Technology Recognition

| Technology | Purpose |
|---|---|
| Llama 3.1 | Large Language Model |
| Gemini | Generative AI model family |
| FAISS | Vector similarity search |
| BM25 | Traditional text retrieval |
| Embeddings | Numerical vector representations |
| RAG | Retrieval-Augmented Generation |
| LangGraph | Agent/workflow orchestration |
| FastAPI | Python API framework |
| Pydantic | Data validation |

## 8. Self-Test — Answer Without Looking

1. What is the difference between AI and ML?
2. Why is DL considered a subset of ML?
3. What is the difference between a feature and a label?
4. Explain supervised, unsupervised and reinforcement learning.
5. What happens during forward propagation?
6. What is the role of backpropagation?
7. What is gradient descent?
8. What is the difference between training and inference?
9. Why is ChatGPT considered Generative AI?
10. Can a bank use ML without GenAI?
11. Why can a rule-based fraud detection system miss suspicious transactions?
12. Why doesn't 99% accuracy necessarily mean a fraud model is good?

## 9. Practical Checklist

- [x] Implemented rule-based fraud detection in Python.
- [x] Implemented a basic Decision Tree classifier.
- [x] Executed both programs successfully.
- [x] Compared rule-based and ML approaches.

**Important:** The Decision Tree exercise used a tiny synthetic dataset. It demonstrates basic ML mechanics, not a production-ready fraud detection system.

## 10. Day 1 Key Takeaway

AI is the broader field. ML learns patterns from data. DL uses multi-layered neural networks. Generative AI creates new content.

The most important distinction: a traditional program follows explicitly written rules, while an ML model learns patterns from examples.