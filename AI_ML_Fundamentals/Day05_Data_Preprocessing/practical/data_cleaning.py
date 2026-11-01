import pandas as pd
import numpy as np

data = {
    "transaction_id": [1, 2, 3, 4, 5, 5, 7, 8],
    "amount": [500, 95000, np.nan, 80000, 120000, 120000, 9000000, 2500],
    "age": [25, 32, 28, np.nan, 41, 41, 45, 22],
    "location": ["Mumbai", "Delhi", "Mumbai", "Pune", "Delhi", "Delhi", "Pune", "Mumbai"],
    "account_type": ["Savings", "Current", "Savings", "Salary", "Current", "Current", "Salary", "Savings"],
    "fraud": [0, 1, 0, 1, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

# Display the dataset information using info().
print(df.info())

# Identify missing values in each column.
print(df.isnull().sum())

# Remove duplicate transactions based on transaction_id.
df = df.drop_duplicates(subset='transaction_id')

# Fill missing amount and age values using their respective medians.
df["amount"] = df["amount"].fillna(df["amount"].median())
df['age'] = df['age'].fillna(df['age'].median())

# Find potential outliers in amount using IQR.
Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["amount"] < lower) |
    (df["amount"] > upper)
]

print(outliers)

# Convert location into one-hot encoded columns.
df = pd.get_dummies(df, columns=['location'])

# Standardize amount using StandardScaler.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df["amount_scaled"] = scaler.fit_transform(df[["amount"]])

# Display the fraud vs legitimate transaction counts.
print(df['fraud'].value_counts())