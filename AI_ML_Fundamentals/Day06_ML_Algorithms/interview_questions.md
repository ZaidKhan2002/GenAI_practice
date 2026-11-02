# Day 6 — Machine Learning Interview Questions

**1. What is the difference between regression and classification?**

Regression predicts continuous numerical values, whereas classification predicts discrete categories.

**2. Why is Logistic Regression used for classification despite its name?**

It applies a sigmoid function to a weighted score to estimate class probabilities, which can be converted into class predictions using a threshold.

**3. What is the sigmoid function?**

It maps a numerical input to a value between 0 and 1, allowing the output to be interpreted as a probability estimate.

**4. What is the difference between `predict()` and `predict_proba()`?**

`predict()` returns predicted class labels. `predict_proba()` returns estimated probabilities for each class.

**5. How does a Decision Tree work?**

It recursively splits data using feature-based conditions, selecting splits that reduce impurity.

**6. What is Gini impurity?**

A measure of how mixed the classes are within a node. Lower Gini impurity indicates a purer node.

**7. What is `max_depth`?**

It limits the maximum depth of a Decision Tree and helps control model complexity.

**8. Why can a Decision Tree overfit?**

A deep tree can learn noise and specific training examples instead of general patterns.

**9. Why does a Decision Tree generally not require feature scaling?**

Its splits depend on feature ordering and thresholds. Scaling usually preserves the ordering and corresponding split structure.

**10. What is the difference between Logistic Regression and Decision Tree?**

Logistic Regression learns a weighted relationship and has a linear decision boundary by default. A Decision Tree learns sequential feature-based splits and can represent nonlinear decision boundaries.

**11. Can Decision Trees perform regression?**

Yes. DecisionTreeRegressor predicts numerical values, while DecisionTreeClassifier predicts categories.

**12. Why is 100% accuracy not necessarily proof of a good model?**

The test set may be too small, unrepresentative or affected by leakage. A model must be evaluated on realistic unseen data.