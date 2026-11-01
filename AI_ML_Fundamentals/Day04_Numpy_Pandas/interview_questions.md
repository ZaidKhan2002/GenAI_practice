# Day 4 — Interview Questions

## NumPy

**1. What is NumPy?**

NumPy is a Python library used for efficient numerical computation and array operations.

**2. Difference between Python lists and NumPy arrays?**

Python lists are flexible and can store mixed data types. NumPy arrays generally use a common data type and support efficient numerical operations.

**3. What is vectorization?**

Performing operations on entire arrays without explicitly writing Python loops.

**4. What does the shape of a NumPy array represent?**

It represents the dimensions of the array. For example, `(3, 2)` means three rows and two columns.

**5. What is the difference between `np.mean()` and `np.sum()`?**

`np.mean()` calculates the average; `np.sum()` calculates the total.

## Pandas

**6. What is Pandas?**

A Python library used for data manipulation, cleaning and analysis.

**7. Difference between Series and DataFrame?**

A Series is one-dimensional; a DataFrame is two-dimensional.

**8. How do you inspect a DataFrame?**

Using `head()`, `info()`, `describe()`, `shape` and `columns`.

**9. How do you filter rows based on multiple conditions?**

Use Boolean conditions with `&` for AND and `|` for OR. Enclose each condition in parentheses.

**10. How do you identify missing values?**

Use `df.isnull().sum()`.

**11. How do you handle missing values?**

Depending on the data, remove affected records, use statistical imputation, apply domain-specific defaults or introduce missing-value indicators.

**12. Why is Pandas useful in ML?**

It makes it easier to load, inspect, clean, transform and prepare datasets before model training.

## Practical interview exercise

Given a DataFrame containing transaction amounts and fraud labels:

- Find transactions above ₹50,000.
- Calculate average transaction amount.
- Count fraudulent records.
- Identify missing values.
- Convert the amount column into a NumPy array.

Be prepared to explain not just the syntax, but why each operation is useful.