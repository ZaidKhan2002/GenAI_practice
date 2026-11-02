# Day 8 — Neural Networks Interview Questions

## Beginner Questions

### 1. What is a neural network?
A machine learning model consisting of interconnected artificial neurons organised into layers that learn patterns from data.

### 2. What are the three main layers?
- Input layer: Receives input features.
- Hidden layers: Learn patterns and representations.
- Output layer: Produces predictions.

### 3. What is an artificial neuron?
A computational unit that calculates a weighted sum of inputs, adds a bias and applies an activation function.

### 4. What are weights and biases?
Weights determine the influence of inputs. Bias shifts the weighted sum before activation. Both are learned during training.

### 5. What is an activation function?
A function applied to a neuron's weighted sum to introduce non-linearity.

## Intermediate Questions

### 6. Why do neural networks need activation functions?
Without non-linear activation functions, stacking linear layers still produces an overall linear transformation. Non-linearity enables learning complex relationships.

### 7. What is ReLU?
ReLU is an activation function defined as:

\[
f(x)=\max(0,x)
\]

It returns zero for negative inputs and the input itself for positive inputs.

### 8. What is the difference between ReLU and Sigmoid?
ReLU has an output range of 0 to infinity, while Sigmoid produces values between 0 and 1.

### 9. What is forward propagation?
The process of passing input data through successive network layers to calculate the final prediction.

### 10. What is the difference between training and inference?
Training learns model parameters by minimising prediction error. Inference uses the learned parameters to generate predictions.

### 11. What is the difference between traditional ML and neural networks?
Traditional ML often relies more heavily on manual feature engineering. Neural networks learn feature representations through multiple layers.

### 12. Why are neural networks useful?
They can learn complex, nonlinear patterns and representations, making them useful in computer vision, NLP, speech and generative AI.

## Practical Questions

### 13. What does np.dot(x, w) do?
Calculates the dot product of two arrays, multiplying corresponding elements and summing the results.

### 14. What does np.maximum(0, z) do?
Applies ReLU element-wise by replacing negative values with zero.

### 15. Are weights manually assigned in a trained neural network?
Weights are typically initialised automatically and subsequently learned through optimisation during training.

### 16. Are weights updated during inference?
No. Standard inference uses the trained parameters without updating them.

## Important Formulas

Weighted sum:

\[
z=\sum_i w_ix_i+b
\]

Neuron output:

\[
a=f(z)
\]

ReLU:

\[
f(x)=\max(0,x)
\]

Sigmoid:

\[
f(x)=\frac{1}{1+e^{-x}}
\]