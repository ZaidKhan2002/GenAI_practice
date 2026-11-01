# Day 5 — Data Preprocessing and EDA

## 1. Data Preprocessing

Data preprocessing transforms raw data into a clean, consistent format suitable for machine learning.

Main activities:
- Missing-value handling.
- Duplicate removal.
- Outlier detection.
- Categorical encoding.
- Feature scaling.

## 2. Missing Values

Identify missing values:

```python
df.isnull().sum()
```

Common approaches:
- `dropna()` — remove rows with missing values.
- `fillna()` — replace missing values.
- Mean — useful for relatively symmetric numerical data.
- Median — more robust to extreme values.
- Mode or an explicit category — often useful for categorical data.

Choose the strategy based on the data and business context.

## 3. Duplicate Records

```python
df.duplicated().sum()

df = df.drop_duplicates(
    subset=["transaction_id"]
)
```

Use an appropriate business key to identify duplicates. Identical-looking records are not necessarily duplicates.

## 4. Outlier Detection

An outlier is an observation unusually far from the rest of the dataset.

IQR method:

```python
Q1 = df["amount"].quantile(0.25)
Q3 = df["amount"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["amount"] < lower) |
    (df["amount"] > upper)
]
```

Important:
- An outlier is not necessarily invalid.
- A genuine rare transaction may be valuable training data.
- Investigate before removing observations.

## 5. Numerical and Categorical Features

**Numerical:** Amount, age, failed login count.

**Categorical:** Location, account type, payment method.

### One-hot encoding

Creates separate binary columns for nominal categories.

```python
df = pd.get_dummies(
    df,
    columns=["location"],
    dtype=int
)
```

### Label / Ordinal encoding

- Label encoding is commonly used for target labels.
- Ordinal encoding represents categories with a genuine order.
- Avoid assigning arbitrary numerical order to nominal input categories.

## 6. Feature Scaling

### Standardization

Transforms features using mean and standard deviation.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df["amount_scaled"] = scaler.fit_transform(
    df[["amount"]]
)
```

### Normalization

Min-Max scaling commonly transforms values into the 0–1 range.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df["amount_normalized"] = scaler.fit_transform(
    df[["amount"]]
)
```

Scaling is generally useful for KNN, SVM, logistic regression and neural networks. It is usually unnecessary for Decision Trees and Random Forests.

## 7. Exploratory Data Analysis (EDA)

EDA means investigating data before model training.

Useful operations:

```python
df.head()
df.info()
df.describe()
df.shape
df.isnull().sum()
df.duplicated().sum()
df["fraud"].value_counts()
```

EDA helps identify:
- Missing values.
- Duplicate records.
- Outliers.
- Feature distributions.
- Relationships between features.
- Class imbalance.

## 8. Class Imbalance

Class imbalance occurs when one class has significantly more observations than another.

Example:
- 9,900 legitimate transactions.
- 100 fraudulent transactions.

A model predicting everything as legitimate achieves 99% accuracy but detects no fraud.

Evaluate precision, recall, F1-score and the confusion matrix as well.

## 9. Data Leakage

Data leakage occurs when information unavailable at prediction time, or information from the test set, influences model training.

Correct approach:

1. Split into training and testing sets.
2. Fit preprocessing transformations on training data.
3. Transform training and testing data using those fitted transformations.
4. Train and evaluate the model.

Scikit-learn `Pipeline` and `ColumnTransformer` help prevent inconsistent preprocessing and leakage.

## 10. Practical completed

Built a transaction preprocessing workflow using synthetic data:
- Inspected the dataset.
- Identified missing values.
- Removed duplicate transaction IDs.
- Applied median imputation.
- Detected potential outliers using IQR.
- Applied one-hot encoding.
- Standardized transaction amounts.
- Analysed fraud distribution.

**Reminder:** The exercise demonstrates preprocessing concepts; it is not a production fraud-detection pipeline.