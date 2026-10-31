# Day 2 — Interview Questions

1. **What is the difference between a feature and a label?**  
   Features are input variables; labels are the expected outputs.

2. **Why do we split a dataset into training and testing sets?**  
   To train on one portion and evaluate performance on unseen data.

3. **What is the purpose of a validation dataset?**  
   To tune hyperparameters and compare models before final testing.

4. **What is overfitting?**  
   When a model learns training data too closely, including noise, and performs poorly on unseen data.

5. **What is underfitting?**  
   When a model is too simple to learn the underlying patterns.

6. **What is generalisation?**  
   A model's ability to perform well on unseen data.

7. **What is a confusion matrix?**  
   A table showing true positives, true negatives, false positives and false negatives.

8. **What is precision?**  
   The proportion of predicted positive cases that are actually positive.

9. **What is recall?**  
   The proportion of actual positive cases correctly identified.

10. **Why can accuracy be misleading in fraud detection?**  
    Fraud cases are often rare. Predicting every transaction as legitimate can give high accuracy while detecting no fraud.

11. **What is the difference between precision and recall?**  
    Precision measures the correctness of positive predictions; recall measures how many actual positive cases are detected.

12. **What is Gini Impurity?**  
    A measure of class mixing used by Decision Trees to evaluate potential splits.

13. **Why should we not repeatedly tune using the test dataset?**  
    It causes information leakage from the test set and makes the final evaluation unreliable.

14. **How can we reduce overfitting in a Decision Tree?**  
    Limit tree depth, use appropriate regularisation and validate using unseen data.

15. **What does an F1-score represent?**  
    The harmonic mean of precision and recall, useful when both matter.