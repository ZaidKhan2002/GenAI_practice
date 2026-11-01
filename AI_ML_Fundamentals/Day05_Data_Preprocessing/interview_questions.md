# Day 5 — Interview Questions

**1. What is data preprocessing?**

Cleaning and transforming raw data into a suitable format for machine learning.

**2. Mean vs median imputation?**

Mean is sensitive to extreme values. Median is more robust and is often preferable for skewed numerical features.

**3. How do you identify duplicates in Pandas?**

Use `duplicated()` and remove them with `drop_duplicates()`.

**4. What is an outlier?**

An observation unusually far from the rest of the dataset. It may be valid or invalid.

**5. Explain the IQR method.**

IQR is Q3 minus Q1. Potential outliers lie below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR.

**6. Label encoding vs one-hot encoding?**

Label encoding assigns numerical labels. One-hot encoding creates separate binary columns for categories.

**7. What is feature scaling?**

Transforming numerical features to comparable scales.

**8. Standardization vs normalization?**

Standardization uses mean and standard deviation. Min-Max normalization commonly maps values to the 0–1 range.

**9. Why does KNN benefit from scaling?**

KNN calculates distances, so features with larger numerical ranges can dominate distance calculations.

**10. Why don't Decision Trees generally need scaling?**

They use threshold-based splits rather than distance calculations.

**11. What is EDA?**

Exploratory Data Analysis: examining data structure, distributions, missing values, outliers and relationships before modelling.

**12. What is class imbalance?**

When one target class has significantly more observations than another.

**13. Why can accuracy be misleading?**

A model can predict only the majority class and still achieve high accuracy while completely missing the minority class.

**14. What is data leakage?**

When information from outside the legitimate training process influences model training, producing overly optimistic evaluation results.

**15. How do you prevent data leakage?**

Split data first, fit preprocessing transformations on training data only, then apply those transformations to validation and test data.

## Interview preparation tip

For every preprocessing decision, explain:
- What problem are you solving?
- Why did you choose this technique?
- What alternatives exist?
- What could go wrong?