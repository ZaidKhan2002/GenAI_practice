# Day 5 — Quick Revision

## Data Cleaning

| Task | Pandas operation |
|---|---|
| Find missing values | `df.isnull().sum()` |
| Remove missing rows | `df.dropna()` |
| Fill missing values | `df.fillna()` |
| Find duplicates | `df.duplicated()` |
| Remove duplicates | `df.drop_duplicates()` |

## Outliers

IQR = Q3 − Q1

- Lower boundary = Q1 − 1.5 × IQR
- Upper boundary = Q3 + 1.5 × IQR

Never automatically remove an outlier.

## Encoding

- One-hot encoding: nominal categories.
- Ordinal encoding: ordered categories.
- Label encoding: commonly used for target labels.

## Feature Scaling

- StandardScaler: mean 0, standard deviation 1.
- MinMaxScaler: commonly maps values to 0–1.

Scaling generally matters for KNN, SVM and gradient-based models.

## EDA

Remember:

```python
df.head()
df.info()
df.describe()
df.shape
df.isnull().sum()
df["fraud"].value_counts()
```

## Data Leakage

**Split first → Fit on training data → Transform test data → Evaluate.**

Never use test-set information to fit preprocessing transformations.

## Day 5 Practical Results

| Metric | Result |
|---|---:|
| Records after deduplication | 7 |
| Imputed amount | ₹87,500 |
| Imputed age | 30 |
| Potential outlier | ₹90,00,000 |
| Fraud records | 2 |
| Legitimate records | 5 |

Dataset: synthetic transaction data.

## Five things to remember

1. Clean data before training.
2. Median is more robust to extreme values than mean.
3. Outlier does not mean invalid record.
4. Scaling depends on the ML algorithm.
5. Prevent data leakage by fitting preprocessing on training data only.