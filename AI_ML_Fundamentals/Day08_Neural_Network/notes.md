# Day 8 — Neural Networks

## 1. What is a Neural Network?

A neural network is a machine learning model inspired by the structure of biological neurons. It consists of interconnected artificial neurons organised into layers.

Neural networks learn patterns and relationships from data by adjusting weights and biases during training.

They are the foundation of modern deep learning and LLMs.

## 2. Biological vs Artificial Neuron

An artificial neuron:
1. Receives numerical inputs.
2. Multiplies inputs by their corresponding weights.
3. Adds a bias.
4. Applies an activation function.
5. Produces an output.

### Mathematical representation

\[
z = \sum_{i=1}^{n} w_i x_i + b
\]

\[
a = f(z)
\]

Where:
- \(x\): Input features
- \(w\): Weights
- \(b\): Bias
- \(z\): Weighted sum plus bias
- \(f\): Activation function
- \(a\): Neuron output

## 3. Neural Network Architecture

A neural network generally contains three types of layers.

### Input Layer
Receives input features.

Example: Transaction amount and failed login attempts.

### Hidden Layers
Process inputs and learn complex patterns through interconnected neurons.

Multiple hidden layers are a key characteristic of deep learning networks.

### Output Layer
Produces the final prediction.

Example: Fraud probability = 0.92.

## 4. Weights and Biases

**Weights**
- Determine how strongly an input influences a neuron.
- Learned during training.
- Can be positive or negative.

**Bias**
- An additional learnable parameter.
- Shifts the weighted sum before activation.
- Helps neurons learn flexible decision boundaries.

## 5. Activation Functions

Activation functions introduce non-linearity, allowing neural networks to learn complex relationships.

### ReLU

\[
f(x) = \max(0,x)
\]

- Negative inputs become zero.
- Positive inputs remain unchanged.
- Commonly used in hidden layers.

### Sigmoid

\[
f(x)=\frac{1}{1+e^{-x}}
\]

- Output range: 0 to 1.
- Commonly used in binary classification output layers.

### Tanh

\[
f(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
\]

- Output range: -1 to 1.
- Zero-centred activation function.

Other activation functions include GELU and SiLU.

## 6. Forward Propagation

Forward propagation is the process of passing input data through the network to generate a prediction.

Steps:
1. Receive input features.
2. Calculate weighted sums.
3. Add biases.
4. Apply activation functions.
5. Pass outputs to subsequent layers.
6. Generate the final prediction.

## 7. Training vs Inference

### Training
- Model receives training data.
- Forward propagation generates predictions.
- Loss function measures prediction error.
- Backpropagation calculates gradients.
- Optimiser updates weights and biases.
- Process repeats across training iterations.

### Inference
- Receives new input data.
- Uses learned weights and biases.
- Performs forward propagation.
- Produces predictions.
- Does not normally update model parameters.

## 8. Traditional ML vs Neural Networks

Traditional ML often relies more on manually engineered features.

Neural networks can automatically learn useful feature representations through multiple layers.

Both learn from training data, but their learning mechanisms and architectures differ.

## 9. Practical Implementation

Implemented a single artificial neuron using NumPy.

```python
import numpy as np

x = np.array([0.8, 0.6])
w = np.array([0.5, 0.9])
b = -0.4

weighted_sum = np.dot(x, w)
z = weighted_sum + b
final_output = np.maximum(0, z)

print("Final Output:", final_output)
```

Expected output:

```text
Final Output: 0.54
```

This implements forward calculation for one neuron using fixed parameters. It does not implement model training or parameter updates.

## 10. Connection to GenAI

Neural networks form the foundation of:
- Deep learning
- Computer vision
- Speech recognition
- Transformer architectures
- Large Language Models
- Generative AI systems

Important learning path:

Neural Networks → Deep Learning → Transformers → LLMs → GenAI Applications