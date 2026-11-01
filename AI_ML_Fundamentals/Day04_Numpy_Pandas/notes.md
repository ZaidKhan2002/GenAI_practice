# Day 4 — NumPy and Pandas

## 1. NumPy

NumPy is a Python library for efficient numerical computing.

### NumPy arrays
- Store elements in an array, generally with a common data type.
- Support efficient mathematical operations.
- Are less flexible than Python lists.
- Operations such as `np.append()` create a new array.

```python
import numpy as np

arr = np.array([10, 20, 30])

print(arr.shape)
print(arr.dtype)
print(np.mean(arr))
print(np.max(arr))
print(np.sum(arr))
```

### Vectorization
Vectorization performs operations across array elements without explicitly writing Python loops.

```python
arr = np.array([10, 20, 30])
print(arr * 2)
```

### Important concepts
- `shape`: dimensions of an array.
- `dtype`: underlying data type.
- Indexing: access individual elements.
- Slicing: access a range or subset of elements.

## 2. Pandas

Pandas is a Python library for data manipulation and analysis.

### Series vs DataFrame
- Series: one-dimensional labelled data.
- DataFrame: two-dimensional tabular data with rows and columns.

```python
import pandas as pd

data = {
    "amount": [500, 95000, 1000],
    "fraud": [0, 1, 0]
}

df = pd.DataFrame(data)
```

### Important operations

| Operation | Purpose |
|---|---|
| `df.head()` | First five rows |
| `df.tail()` | Last five rows |
| `df.shape` | Number of rows and columns |
| `df.columns` | Column names |
| `df.info()` | Structure and data types |
| `df.describe()` | Statistical summary |
| `df.isnull().sum()` | Missing values per column |
| `df.value_counts()` | Frequency of values |

### Filtering

```python
df[df["amount"] > 50000]
```

Multiple conditions:

```python
df[
    (df["amount"] > 50000) &
    (df["failed_logins"] >= 3)
]
```

### Missing values

```python
df.dropna()
df["amount"].fillna(df["amount"].median())
```

Choose a missing-value strategy based on the data. Fit imputation values using training data only to prevent data leakage.

## 3. NumPy and Pandas in ML

Typical workflow:

1. Load dataset using Pandas.
2. Inspect and understand the data.
3. Clean missing or invalid values.
4. Filter and transform columns.
5. Convert data into numerical arrays when needed.
6. Pass features and labels to an ML model.

## 4. Practical completed

Implemented a transaction analyzer that:
- Created a DataFrame.
- Inspected rows and dimensions.
- Calculated average transaction amount.
- Filtered high-value and high-risk transactions.
- Counted fraud and legitimate records.
- Converted a Pandas column into a NumPy array.
- Calculated maximum and total transaction amounts.

**Important:** The exercise uses a small synthetic dataset, not real fraud-detection data.