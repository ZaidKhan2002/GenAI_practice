# Day 6 — Quick Revision

## Regression vs Classification
- Regression → numerical output.
- Classification → category.

## Linear Regression
- Equation: `y = mx + c`
- Learns slope and intercept.
- Common loss: MSE.
- Used for continuous predictions.

## Logistic Regression
- Weighted score → sigmoid → probability.
- Default binary threshold commonly 0.5.
- `predict()` → class.
- `predict_proba()` → probabilities.

## Decision Tree
- Learns feature-based splits.
- Gini impurity measures class mixing.
- `max_depth` controls complexity.
- Deep trees can overfit.
- Usually no scaling required.

## Evaluation
- Accuracy → overall correctness.
- Precision → how many predicted positives were correct.
- Recall → how many actual positives were found.
- F1 → balance between precision and recall.
- Confusion matrix → TN, FP, FN, TP.

## Remember
- High accuracy does not guarantee a useful model.
- Use appropriate metrics for the business problem.
- Avoid data leakage.
- Evaluate on representative unseen data.
- Understand the algorithm's limitations before choosing it.

## Practical files
- `linear_regression.py`
- `logistic_regression.py`
- `decision_tree.py`