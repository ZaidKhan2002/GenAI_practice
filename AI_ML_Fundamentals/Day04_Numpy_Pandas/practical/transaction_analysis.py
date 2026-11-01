import numpy as np
import pandas as pd

data = {
    "transaction_id": [1, 2, 3, 4, 5, 6],
    "amount": [500, 95000, 1000, 80000, 120000, 2500],
    "failed_logins": [0, 5, 0, 4, 2, 1],
    "fraud": [0, 1, 0, 1, 0, 0]
}

df = pd.DataFrame(data)

# 1. Display the complete DataFrame
print(df)

# 2. Display the first 3 rows
print(df.head(3))

# 3. Print the shape of the DataFrame
print(df.shape)

# 4. Calculate the average transaction amount
average_amount = df['amount'].mean()
print(f"Average transaction amount: {average_amount}")

# 5. Filter transactions where amount > 50000
filtered_transactions = df[df['amount'] > 50000]
print(filtered_transactions)

# 6. Filter transactions where amount > 50000
#    AND failed_logins >= 3
filtered_transactions_2 = df[(df['amount'] > 50000) & (df['failed_logins'] >= 3)]
print(filtered_transactions_2)

# 7. Count fraudulent and legitimate transactions
fraud_count = df['fraud'].value_counts().get(1, 0) 
legitimate_count = df['fraud'].value_counts().get(0,0)
print(fraud_count)
print(legitimate_count)

# 8. Convert the amount column into a NumPy array
amount_array = df['amount'].to_numpy()

# 9. Calculate the maximum transaction amount using NumPy
max_amount = np.max(amount_array)
print(f"Maximum transaction amount: {max_amount}")

# 10. Calculate the total transaction amount using NumPy
total_amount = np.sum(amount_array)
print(f"Total transaction amount: {total_amount}")