# Day 9 — Interview Questions: Deep Learning Training

## Fundamentals

**1. What is a loss function?**

A loss function measures the difference between model predictions and actual targets. Training aims to minimise this loss.

**2. What is the difference between MSE and Binary Cross Entropy?**

MSE is commonly used for regression. Binary Cross Entropy is used for binary classification with probability predictions.

**3. What is backpropagation?**

Backpropagation calculates gradients of the loss with respect to trainable parameters using the chain rule.

**4. What is gradient descent?**

An optimisation algorithm that updates parameters using gradients to reduce loss.

**5. Difference between backpropagation and gradient descent?**

Backpropagation calculates gradients; gradient descent or another optimiser uses them to update parameters.

## Training concepts

**6. What happens if the learning rate is too high?**

Updates may overshoot the minimum, causing oscillation or preventing convergence.

**7. What happens if the learning rate is too low?**

Training progresses very slowly.

**8. Explain epoch, batch and iteration.**

An epoch is one complete pass through training data. A batch is a subset of records. An iteration is one optimiser update.

**9. A dataset contains 1,000 records with batch size 100. How many iterations per epoch?**

10 iterations.

**10. What is an optimiser?**

An optimiser uses gradients to update model parameters and minimise loss.

**11. How does Adam work?**

Adam maintains moving averages of gradients and squared gradients to adapt parameter updates.

**12. Difference between SGD and Adam?**

SGD uses gradient-based updates, while Adam uses adaptive updates based on moving averages of gradients and squared gradients.

## PyTorch

**13. What is a Tensor?**

A PyTorch data structure supporting numerical computation, GPU acceleration and automatic differentiation.

**14. What is Autograd?**

PyTorch's automatic differentiation engine that calculates gradients.

**15. What does `loss.backward()` do?**

Calculates gradients of the loss with respect to trainable parameters and stores them in `.grad`.

**16. What does `optimizer.step()` do?**

Updates model parameters using the calculated gradients.

**17. Why call `optimizer.zero_grad()`?**

PyTorch accumulates gradients by default. This operation clears previous gradients before the next iteration.

**18. What is the purpose of DataLoader?**

It provides data in batches and can shuffle training records.

**19. Difference between `model.train()` and `model.eval()`?**

They set the behaviour of layers such as Dropout and BatchNorm for training and evaluation respectively.

**20. Why use `torch.no_grad()` during inference?**

It disables gradient tracking, reducing unnecessary memory usage and computation.

## Generalisation and troubleshooting

**21. Does low training loss guarantee good performance?**

No. The model may have overfit the training dataset. Evaluate it on unseen data.

**22. What is overfitting?**

When a model learns training-specific patterns too closely and performs poorly on unseen data.

**23. Why standardise features?**

To put numerical features on comparable scales, often improving optimisation stability.

**24. What is the difference between training and inference?**

Training updates model parameters; inference uses learned parameters to generate predictions.

**25. Explain a complete neural network training cycle.**

Input data → forward propagation → loss calculation → backpropagation → optimiser update → repeat across batches and epochs → evaluate on unseen data.