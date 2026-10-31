# Day 1 — AI, ML, Deep Learning and Generative AI

## 1. Artificial Intelligence (AI)
AI is the broader field of building systems capable of performing tasks that typically require human intelligence, such as reasoning, learning, perception, language understanding and decision-making.

Examples:
- Rule-based chatbots
- Fraud detection systems
- Recommendation systems
- Generative AI assistants

## 2. Machine Learning (ML)
ML is a subset of AI in which systems learn patterns from data and use those patterns to make predictions or decisions.

**Traditional programming:**
- Input + explicitly written rules → Output

**Machine Learning:**
- Training data + algorithm → Trained model
- New input + trained model → Prediction

### Important terminology
- Dataset: Collection of examples used for learning.
- Feature: Input characteristic used by a model.
- Label: Known target or expected output.
- Algorithm: Method used to learn patterns.
- Model: Learned representation produced through training.
- Training: Process of learning from data.
- Inference: Using a trained model to generate predictions.

### Types of ML
1. Supervised learning: Learns from labelled examples.
2. Unsupervised learning: Finds patterns in unlabelled data.
3. Reinforcement learning: Learns through actions, rewards and penalties.

## 3. Deep Learning (DL)
DL is a subset of ML that uses neural networks with multiple layers to learn complex patterns.

Examples:
- Image recognition
- Speech recognition
- Language understanding

### Neural network fundamentals
A neuron receives inputs and calculates:

z = w1x1 + w2x2 + b

a = f(z)

Where:
- x = Input feature
- w = Weight
- b = Bias
- f = Activation function
- a = Neuron output

### Neural network training cycle
1. Forward propagation: Calculate predictions.
2. Loss calculation: Measure prediction error.
3. Backpropagation: Calculate gradients.
4. Gradient descent: Update weights and biases.
5. Repeat until the model learns sufficiently well.

## 4. Generative AI
Generative AI refers to systems that generate new content such as text, images, audio, video or code.

Examples:
- ChatGPT
- Image generation systems
- Code generation assistants
- Audio generation systems

### Large Language Models (LLMs)
An LLM is a model trained on large amounts of text to understand and generate language.

Modern autoregressive LLMs generate text by repeatedly predicting the next token based on previous context.

Important: Every LLM is not necessarily the same as every Generative AI system. Image and audio generation systems are also examples of GenAI.

## 5. Conceptual hierarchy

AI ⊃ ML ⊃ DL

Modern deep-learning-based Generative AI is built largely using deep learning, although not every generative technique necessarily follows this exact hierarchy.

## 6. Rule-based systems vs ML

| Rule-based | Machine Learning |
|---|---|
| Explicit conditions | Learns patterns from data |
| Predictable rules | Probabilistic predictions |
| Manual rule updates | Model retraining |
| Easy to audit | Requires model evaluation |

A bank can combine both approaches:
- Rules for mandatory business policies.
- ML for identifying suspicious patterns.
- A decision engine to combine the results.

## 7. Important distinction

Not every component in a GenAI application is AI.

Examples:
- Llama: Language model.
- Embeddings: Vector representations of information.
- FAISS: Vector similarity search library.
- BM25: Traditional information retrieval algorithm.
- RAG: Retrieval-Augmented Generation architecture.
- LangGraph: Workflow and agent orchestration framework.
- FastAPI: Python web framework.
- Pydantic: Data validation library.

## 8. Practical work

Implemented:
- Rule-based fraud detection using Python.
- A basic Decision Tree classifier using scikit-learn.

Note: The ML exercise uses a tiny synthetic dataset for learning. It is not suitable for real fraud detection.

## Key takeaway

AI is the broader field, ML learns from data, DL uses multi-layered neural networks, and Generative AI produces new content.