# Day 2 — Machine Learning Workflow

## 1. Dataset Fundamentals

- **Observation:** One individual record/row.
- **Feature (X):** Input variable used by the model.
- **Label (y):** Expected output the model learns to predict.
- **Dataset:** Collection of observations.

Example: Transaction fraud detection

- Features: Transaction amount, failed login attempts.
- Label: Fraud (1) or Legitimate (0).

## 2. Training, Validation and Testing

| Dataset | Purpose |
|---|---|
| Training | Model learns patterns and parameters. |
| Validation | Tune hyperparameters and compare models. |
| Testing | Evaluate final performance on unseen data. |

An illustrative split is 70% training, 15% validation and 15% testing.

**Important:** Never repeatedly tune a model using the test set. For time-dependent data, chronological splitting may prevent data leakage.

## 3. How Models Learn

Different algorithms learn patterns differently.

- Decision Tree: Learns decision rules by splitting features.
- Linear Regression: Learns coefficients.
- Neural Network: Learns weights and biases.

**Gini Impurity:** Measures how mixed the classes are in a Decision Tree node. Lower impurity means a purer node.

## 4. Underfitting vs Overfitting

| Underfitting | Overfitting |
|---|---|
| Model is too simple. | Model learns training data too closely. |
| Fails to learn important patterns. | Learns noise and specific examples. |
| Poor training and test performance. | Good training, poor test performance. |

- Underfitting: Increase model complexity or improve features.
- Overfitting: Regularisation, limit tree depth, use representative data and cross-validation.
- **Generalisation:** Ability to perform well on unseen data.

## 5. Model Evaluation Metrics

Using fraud as the positive class:

| Metric | Formula | Meaning |
|---|---|---|
| Accuracy | (TP + TN) / Total | Overall correct predictions |
| Precision | TP / (TP + FP) | How many predicted fraud cases were actually fraud |
| Recall | TP / (TP + FN) | How many actual fraud cases were detected |
| F1-score | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | TN | FP |
| Actual Fraud | FN | TP |

- TP: Fraud correctly detected.
- TN: Legitimate correctly classified.
- FP: Legitimate incorrectly flagged as fraud.
- FN: Fraud incorrectly classified as legitimate.

## 6. Class Imbalance

Example: 10,000 transactions containing only 100 fraudulent transactions.

A model predicting every transaction as legitimate achieves 99% accuracy but detects zero fraud cases.

**Takeaway:** Accuracy alone can be misleading. Monitor precision, recall, F1-score and the confusion matrix.

## 7. Practical Implementation

File: `02-ML-Workflow/practical/fraud_classifier.py`

Implemented using:
- `train_test_split()` — split training and testing data.
- `DecisionTreeClassifier()` — train a classification model.
- `fit()` — train the model.
- `predict()` — generate predictions.
- `accuracy_score()`, `precision_score()`, `recall_score()`, `f1_score()` — evaluate performance.
- `confusion_matrix()` — inspect classification errors.

**Practical result:** All metrics were 100%, with confusion matrix `[[3, 0], [0, 2]]`.

This result is not evidence of a production-ready model: the test set contained only five synthetic records.