# Day 9 — Quick Revision

## Remember these distinctions

| Concept | Remember |
|---|---|
| Loss | Measures prediction error |
| Backpropagation | Calculates gradients |
| Optimiser | Updates parameters |
| Learning rate | Controls update size |
| Epoch | One full dataset pass |
| Batch | Subset of records |
| Iteration | One optimiser update |
| Adam | Adaptive optimiser |
| Tensor | PyTorch computation structure |
| Autograd | Automatic gradient calculation |

## PyTorch training sequence

```python
optimizer.zero_grad()

predictions = model(inputs)

loss = criterion(predictions, labels)

loss.backward()

optimizer.step()
```

## Key formulas

Gradient descent:

`w_new = w_old - learning_rate × gradient`

MSE:

`MSE = (1/n) × Σ(y - ŷ)²`

Binary Cross Entropy:

`L = -[y log(p) + (1-y) log(1-p)]`

## PyTorch inference

```python
model.eval()

with torch.no_grad():
    predictions = model(inputs)
```

## Common interview traps

1. Backpropagation does not update weights; the optimiser does.
2. An epoch does not mean all records are processed in one batch.
3. Low training loss does not guarantee generalisation.
4. `model.eval()` does not disable gradient tracking.
5. `optimizer.zero_grad()` matters because gradients accumulate.
6. Adam is an optimiser, not a loss function.
7. Inference does not normally update model parameters.

## Practical completed

- Defined a neural network using `nn.Module`.
- Used Linear, ReLU and Sigmoid layers.
- Prepared data using StandardScaler.
- Converted data into PyTorch tensors.
- Created batches with DataLoader.
- Trained using BCELoss and Adam.
- Implemented backpropagation and parameter updates.
- Generated predictions using evaluation mode.

**Practical limitation:** Training and predictions were tested on 12 synthetic records. Independent test-set evaluation remains to be done.
