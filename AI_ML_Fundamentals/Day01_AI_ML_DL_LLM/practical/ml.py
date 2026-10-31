from sklearn.tree import DecisionTreeClassifier

# Features:
# [transaction_amount, failed_logins]

X = [
    [500, 0],
    [1000, 0],
    [95000, 5],
    [80000, 4],
    [700, 1],
    [120000, 0],
    [110000, 5],
    [2000, 0]
]

# Labels:
# 0 = Legitimate
# 1 = Fraud

y = [0, 0, 1, 1, 0, 1, 1, 0]

# Create model
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(X, y)

# Predict on new transactions
new_transaction = [[90000, 4]]

prediction = model.predict(new_transaction)

if prediction[0] == 1:
    print("Fraud")
else:
    print("Legitimate")