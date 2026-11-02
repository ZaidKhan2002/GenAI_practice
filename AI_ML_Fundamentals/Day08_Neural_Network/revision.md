# Day 8 — Quick Revision

## Neural Network in One Sentence

A neural network is a machine learning model made of interconnected neurons that learn patterns by adjusting weights and biases.

## Architecture

| Layer | Purpose |
|---|---|
| Input | Receives features |
| Hidden | Learns patterns |
| Output | Produces prediction |

## Artificial Neuron

\[
z=\sum_i w_ix_i+b
\]

\[
a=f(z)
\]

- Input: Data received by the neuron.
- Weight: Importance of an input.
- Bias: Shifts the weighted sum.
- Activation: Introduces non-linearity.
- Output: Result passed to the next layer.

## Activation Functions

| Function | Range | Common use |
|---|---|---|
| ReLU | 0 to infinity | Hidden layers |
| Sigmoid | 0 to 1 | Binary output |
| Tanh | -1 to 1 | Some hidden layers |

## Forward Propagation

Input → Weighted sum → Bias → Activation → Next layer → Prediction.

## Training vs Inference

| Training | Inference |
|---|---|
| Learns parameters | Uses learned parameters |
| Calculates loss | Generates prediction |
| Backpropagates gradients | No normal parameter updates |
| Updates weights and biases | Keeps parameters fixed |

## Important Distinctions

1. Neural networks are a type of machine learning model.
2. Deep learning uses neural networks with multiple layers.
3. Weights determine input influence.
4. Bias shifts the weighted sum.
5. Activation functions introduce non-linearity.
6. Forward propagation calculates predictions.
7. Training updates parameters; inference uses them.
8. A single neuron calculation is not a complete training process.

## Practical Result

Inputs: [0.8, 0.6]

Weights: [0.5, 0.9]

Bias: -0.4

Weighted sum: 0.94

After bias: 0.54

After ReLU: 0.54

**Final output: 0.54**

## What Comes Next?

Day 9 — Training Deep Learning Models:
- Loss functions
- Gradient descent
- Backpropagation
- Learning rate
- Epochs and batches
- Optimisers

**Most important concept to remember:** A neural network does not magically know the correct weights. It learns them by repeatedly measuring prediction error and updating its parameters.