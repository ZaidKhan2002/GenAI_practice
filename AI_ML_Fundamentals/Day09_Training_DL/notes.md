# Day 9 — Training Deep Learning Models

## 1. Why do neural networks need training?

A neural network learns by adjusting its weights and biases to reduce prediction errors.

The training process consists of:
1. Forward propagation
2. Loss calculation
3. Backpropagation
4. Optimiser update
5. Repetition across batches and epochs

## 2. Loss functions

A loss function measures the difference between model predictions and actual targets.

- **MSE (Mean Squared Error):** Commonly used for regression.

  `MSE = (1/n) × Σ(y - ŷ)²`

- **Binary Cross Entropy (BCE):** Used for binary classification when predictions are probabilities.

  `L = -[y log(p) + (1-y) log(1-p)]`

Lower loss generally indicates predictions closer to the targets.

Loss functions guide training. Accuracy, precision and recall are evaluation metrics.

## 3. Backpropagation and Gradient Descent

**Backpropagation**
- Calculates gradients of loss with respect to model parameters.
- Uses the chain rule to propagate gradients backwards through the network.
- PyTorch: `loss.backward()`

**Gradient Descent**
- Uses gradients to update weights and biases.
- Moves parameters in the direction that reduces loss.

`w_new = w_old - learning_rate × gradient`

Remember: Backpropagation calculates gradients; the optimiser updates parameters.

## 4. Learning Rate

Controls the size of parameter updates.

- Too low: training becomes slow.
- Too high: updates may overshoot the minimum or prevent convergence.
- Appropriate: more stable optimisation.

There is no universally correct learning rate.

## 5. Epoch, Batch and Iteration

- **Epoch:** One complete pass through the training dataset.
- **Batch:** A subset of training records processed together.
- **Iteration:** One optimiser update.

Example: 1,000 records and batch size 100 = 10 iterations per epoch. Ten epochs = 100 iterations.

Common approaches:
- Batch Gradient Descent: entire dataset per update.
- Stochastic Gradient Descent: one example per update.
- Mini-batch Gradient Descent: a small group per update; widely used in deep learning.

## 6. Optimisers

Optimisers use calculated gradients to update model parameters.

- SGD: Basic gradient-based updates.
- Adam: Uses moving averages of gradients and squared gradients for adaptive updates.
- AdamW: Adam with decoupled weight decay.

## 7. PyTorch Fundamentals

**Tensor:** PyTorch data structure supporting CPU/GPU computation and automatic differentiation.

**Autograd:** Automatically calculates gradients.

**nn.Module:** Base class for defining neural network models.

Important operations:

| Operation | Purpose |
|---|---|
| `optimizer.zero_grad()` | Clears accumulated gradients |
| `model(inputs)` | Forward propagation |
| `criterion(predictions, labels)` | Calculates loss |
| `loss.backward()` | Calculates gradients |
| `optimizer.step()` | Updates parameters |

## 8. Neural Network Training Pipeline

1. Prepare input features and labels.
2. Perform forward propagation.
3. Calculate loss.
4. Run backpropagation.
5. Update parameters using an optimiser.
6. Repeat for all batches and epochs.
7. Evaluate on unseen data.

## 9. Training vs Inference

- Training: Model learns by updating weights and biases.
- Inference: Model uses learned parameters to make predictions.

Useful PyTorch methods:

- `model.train()`: Sets training behaviour for layers such as Dropout and BatchNorm.
- `model.eval()`: Sets evaluation behaviour for those layers.
- `torch.no_grad()`: Disables gradient tracking.
- `torch.inference_mode()`: Provides a more restrictive inference context that can further reduce overhead.

`model.eval()` does not disable gradient tracking by itself.

## 10. Overfitting

A model can achieve very low training loss but perform poorly on unseen data.

This indicates possible overfitting.

Always evaluate generalisation using separate validation and test data.

## 11. Practical Implementation

Built a binary fraud classification neural network using PyTorch.

Architecture:
- Input: 2 features (amount, failed logins).
- Hidden layer: 4 neurons with ReLU.
- Output: 1 neuron with Sigmoid.

Components:
- StandardScaler for feature scaling.
- TensorDataset and DataLoader for batches.
- BCELoss for binary classification.
- Adam optimiser.
- 100 training epochs and batch size 4.
- `loss.backward()` and `optimizer.step()` for learning.
- `model.eval()` and `torch.no_grad()` for inference.

The model's training predictions matched all 12 synthetic training labels. This demonstrates a working training pipeline, not proven performance on unseen transactions.

**Important:** This exercise used a small synthetic dataset. It is not a production fraud detection model.