# Day 2 — Quick Revision

## Remember These

- **Feature (X):** Input.
- **Label (y):** Expected output.
- **Training:** Learn patterns.
- **Validation:** Tune model.
- **Testing:** Evaluate on unseen data.
- **Generalisation:** Perform well on unseen data.

## Underfitting vs Overfitting

- Underfitting = Model too simple.
- Overfitting = Model memorises training-specific patterns.
- Good generalisation = Performs well on unseen data.

## Evaluation Metrics

- Accuracy = Overall correctness.
- Precision = Correctness of positive predictions.
- Recall = Coverage of actual positive cases.
- F1 = Balance between precision and recall.

## Confusion Matrix

- TP = Correct positive.
- TN = Correct negative.
- FP = False alarm.
- FN = Missed positive.

## Fraud Detection

- Imbalanced data makes accuracy unreliable.
- A model can achieve 99% accuracy and still detect zero fraud.
- Monitor precision, recall, F1 and confusion matrix.

## Practical

Implemented a Decision Tree fraud classifier using Scikit-learn.

Result: 100% across metrics on a five-record synthetic test set. Not sufficient evidence of real-world model performance.

## Key Interview Takeaway

A model is not good merely because it achieves high accuracy. It must generalise to unseen, representative data and perform well on the metrics relevant to the business problem.