import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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

# Separate features (X) and target (y).
X = df[["amount", "failed_logins"]]
y = df["fraud"] 

# Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Create a Decision Tree Classifier model.
model = DecisionTreeClassifier(max_depth=3, random_state=42)

# Train the model using .fit().
model.fit(X_train, y_train)

# Predict classes using .predict().
predictions = model.predict(X_test) 

# Evaluate using accuracy, precision, recall, F1 and confusion matrix.
print("Predictions:", predictions)
print("Accuracy:", accuracy_score(y_test, predictions))

print("Precision:", precision_score(y_test, predictions))

print("Recall:", recall_score(y_test, predictions))

print("F1 Score:", f1_score(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))