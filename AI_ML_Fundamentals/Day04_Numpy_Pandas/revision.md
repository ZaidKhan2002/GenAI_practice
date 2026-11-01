# Day 4 — Quick Revision

## NumPy
- Numerical computing library.
- Array-based operations.
- Efficient mathematical computation.
- Vectorization avoids explicit Python loops.
- `shape` = dimensions.
- `dtype` = data type.
- `np.mean()`, `np.sum()`, `np.max()`, `np.min()`.

## Pandas
- Data manipulation and analysis.
- Series = 1D.
- DataFrame = 2D.
- `head()` = preview rows.
- `info()` = structure and types.
- `describe()` = statistical summary.
- `shape` = dimensions.
- `value_counts()` = frequency distribution.
- `isnull().sum()` = missing-value count.
- `fillna()` = replace missing values.
- `dropna()` = remove missing-value rows.

## Filtering

```python
df[df["amount"] > 50000]

df[
    (df["amount"] > 50000) &
    (df["failed_logins"] >= 3)
]
```

## Remember
- NumPy is primarily for numerical arrays.
- Pandas is primarily for structured/tabular data.
- Both are widely used in ML preprocessing.
- Missing-value treatment depends on the dataset.
- Avoid data leakage when preprocessing training and test data.

## Practical results

| Metric | Result |
|---|---:|
| Dataset rows | 6 |
| Dataset columns | 4 |
| Average amount | 49,833.33 |
| Fraud records | 2 |
| Legitimate records | 4 |
| Maximum amount | 120,000 |
| Total amount | 299,000 |

Dataset: synthetic transaction data.