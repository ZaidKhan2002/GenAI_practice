# Day 6 — Machine Learning Algorithms

## 1. Regression vs Classification

- **Regression:** predicts a continuous numerical value. Example: predicting transaction amount or house price.
- **Classification:** predicts a category. Example: legitimate vs fraudulent transaction.

## 2. Linear Regression

Used to predict continuous numerical values.

**Equation:**

`y = mx + c`

- `y`: predicted output
- `x`: input feature
- `m`: coefficient/slope
- `c`: intercept

Training learns the coefficient and intercept by minimising prediction error, commonly Mean Squared Error (MSE).

**Important:** Predictions outside the training data range are extrapolations and may be unreliable.

Scikit-learn:
- `LinearRegression()`
- `.fit(X, y)`
- `.predict(X)`
- `.coef_`
- `.intercept_`

## 3. Logistic Regression

Despite its name, primarily used for classification.

It calculates a weighted score:

`z = w1*x1 + w2*x2 + ... + b`

The sigmoid function converts this score into a probability:

`σ(z) = 1 / (1 + e^(-z))`

A threshold (commonly 0.5) determines the predicted class.

- `.predict()` returns class labels.
- `.predict_proba()` returns class probabilities.
- `model.classes_` shows the probability-column order.

Lowering the classification threshold can increase recall but may also increase false positives.

## 4. Decision Tree

A model that learns a sequence of feature-based decisions.

Example:
- Is amount greater than ₹50,000?
- Are failed logins greater than or equal to 3?
- Predict a class based on the resulting path.

The tree selects splits that reduce impurity.

### Gini Impurity

`Gini = 1 - Σ(p_i²)`

- 0 means a completely pure node.
- For two equally represented classes, Gini = 0.5.

Other split criteria include entropy/information gain.

### Controlling complexity

- `max_depth`: maximum tree depth.
- `min_samples_split`: minimum samples required to split a node.
- `min_samples_leaf`: minimum samples required in a leaf.

A very deep tree can overfit.

## 5. Algorithm comparison

| Algorithm | Primary use | Scaling |
|---|---|---|
| Linear Regression | Numerical prediction | Depends on features and use case |
| Logistic Regression | Classification | Often useful |
| Decision Tree | Classification and regression | Generally unnecessary |

## 6. Evaluation metrics

- Accuracy = (TP + TN) / Total
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × (Precision × Recall) / (Precision + Recall)

Confusion matrix:

`[[TN, FP], [FN, TP]]`

For fraud detection, accuracy alone can be misleading, particularly when fraud cases are rare.

## 7. Practical work completed

Implemented three algorithms using Python and scikit-learn:

1. Linear Regression — predicted productivity score from training hours.
2. Logistic Regression — classified synthetic transactions.
3. Decision Tree — classified the same synthetic transactions.

Both classification models achieved 100% on five synthetic test records. This is only a workflow exercise, not evidence of production performance.

## Key takeaway

Choose an algorithm based on the problem, data, evaluation metrics, interpretability and operational constraints—not simply because it achieves high accuracy on a small test set.