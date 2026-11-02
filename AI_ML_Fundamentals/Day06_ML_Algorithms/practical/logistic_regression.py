import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score
)

data = {
    "amount": [
        500, 1000, 2000, 1500,
        95000, 80000, 70000, 120000,
        3000, 5000, 90000, 110000,
        800, 2500, 60000, 75000,
        4000, 10000, 85000, 130000
    ],
    "failed_logins": [
        0, 0, 1, 0,
        5, 4, 3, 5,
        0, 1, 4, 6,
        0, 0, 3, 5,
        1, 0, 4, 6
    ],
    "fraud": [
        0, 0, 0, 0,
        1, 1, 1, 1,
        0, 0, 1, 1,
        0, 0, 1, 1,
        0, 0, 1, 1
    ]
}

df = pd.DataFrame(data)

# Separate features (X) and target (y)
X = df[["amount", "failed_logins"]]
y = df["fraud"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Create a Logistic Regression model
model = LogisticRegression()

# Train it using .fit()
model.fit(X_train, y_train)

# Predict classes using .predict()
predictions = model.predict(X_test)

# Predict probabilities using .predict_proba()
probabilities = model.predict_proba(X_test)

# Print accuracy, precision, recall and confusion matrix.
print("Predictions:", predictions)
print("Probabilities:", probabilities)

print("Accuracy:", accuracy_score(y_test, predictions))

print("Precision:", precision_score(y_test, predictions))

print("Recall:", recall_score(y_test, predictions))

print("F1 Score:", f1_score(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))