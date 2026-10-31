# Day 1 — AI/ML Interview Questions

## Section 1: Core Fundamentals

### Q1. What is Artificial Intelligence?
AI is the broader field of building systems capable of performing tasks that typically require human intelligence, such as reasoning, learning, perception and decision-making.

### Q2. What is Machine Learning?
Machine Learning is a subset of AI in which systems learn patterns from data and use those patterns to make predictions or decisions without relying entirely on explicitly programmed rules.

### Q3. What is the difference between AI and ML?

| AI | ML |
|---|---|
| Broader field | Subset of AI |
| Focuses on intelligent behaviour | Focuses on learning patterns from data |
| Can use rules or learned models | Learns from data |

### Q4. What is Deep Learning?
Deep Learning is a subset of ML that uses neural networks with multiple layers to learn complex patterns from data.

### Q5. What is the difference between ML and DL?
Traditional ML algorithms often rely on manually selected features and relatively simpler models. Deep Learning uses multi-layered neural networks to learn increasingly complex representations from data.

## Section 2: Machine Learning

### Q6. What is a dataset?
A dataset is a collection of examples used to train, validate or test a machine-learning model.

### Q7. What are features and labels?
- Features are the input characteristics used by a model.
- Labels are the known target values the model is trained to predict.

Example:
- Features: Transaction amount, failed login count.
- Label: Fraud or Legitimate.

### Q8. What are the three major types of Machine Learning?

1. **Supervised Learning:** Learns from labelled data. Example: Spam classification.
2. **Unsupervised Learning:** Finds patterns in unlabelled data. Example: Customer segmentation.
3. **Reinforcement Learning:** Learns through actions and rewards. Example: Game-playing agents.

### Q9. What is the difference between training and inference?
- Training: The model learns patterns and parameters from data.
- Inference: A trained model processes new input and generates predictions.

### Q10. Can an organisation use AI without Generative AI?
Yes. Traditional ML can support fraud detection, credit-risk prediction, anomaly detection and demand forecasting. Rule-based systems can also automate decisions without using GenAI.

## Section 3: Deep Learning

### Q11. What is a neural network?
A neural network is a computational model inspired by interconnected neurons. It consists of layers that transform inputs into outputs using learned weights, biases and activation functions.

### Q12. What are weights and biases?
- Weights determine the influence of individual inputs.
- Bias allows a neuron to shift its activation independently of the input values.

### Q13. What is an activation function?
An activation function transforms a neuron's weighted input into its output and introduces non-linearity into the network.

Examples: ReLU, Sigmoid and Tanh.

### Q14. Explain forward propagation and backpropagation.
- Forward propagation calculates the model's prediction by passing inputs through the network.
- Backpropagation calculates gradients of the loss with respect to model parameters so they can be updated.

### Q15. What is gradient descent?
Gradient descent is an optimization algorithm that updates model parameters in the direction that reduces the loss.

## Section 4: Generative AI

### Q16. What is Generative AI?
Generative AI refers to AI systems that generate new content such as text, images, audio, video and code.

### Q17. What is an LLM?
A Large Language Model is a model trained on large amounts of text to understand and generate language.

### Q18. How does an autoregressive LLM generate text?
It predicts a probability distribution over possible next tokens, selects a token according to its decoding strategy, and repeats the process using the updated context.

### Q19. Is every Generative AI system an LLM?
No. LLMs generate language, while other generative models can generate images, audio or video.

### Q20. Is every component in a GenAI application itself AI?
No.

Examples:
- Llama: Language model.
- FAISS: Vector similarity search library.
- BM25: Information retrieval algorithm.
- FastAPI: Backend web framework.
- Pydantic: Data validation library.
- LangGraph: Workflow and agent orchestration framework.

## Section 5: Practical and Scenario-Based Questions

### Q21. A bank uses an explicit condition to flag transactions above ₹100,000. Is this ML?
No. This is traditional rule-based programming because the developer explicitly defines the decision condition.

### Q22. Why might ML identify fraud that a rule-based system misses?
ML can learn combinations and relationships between multiple features from historical examples. It may identify suspicious patterns that were not explicitly encoded as individual rules.

However, ML does not guarantee detection of previously unseen fraud patterns.

### Q23. Why might a bank use both rule-based systems and ML?
Rules can enforce mandatory business policies, while ML can identify complex patterns and produce risk scores. A decision engine can combine both.

### Q24. Does 99% model accuracy necessarily mean a fraud detection model is good?
No. Fraud datasets are often highly imbalanced. A model that predicts every transaction as legitimate could achieve high accuracy while detecting almost no fraud. Precision, recall and F1-score are also important.

## Interview Preparation Reminder

For every answer, be prepared to explain:
1. What the concept means.
2. How it works internally.
3. A practical example.
4. Its limitations and trade-offs.