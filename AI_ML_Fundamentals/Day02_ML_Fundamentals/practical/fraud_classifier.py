from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Synthetic transaction dataset
X = [
    [500, 0], [1000, 0], [2000, 1], [1500, 0],
    [95000, 5], [80000, 4], [70000, 3], [120000, 5],
    [3000, 0], [5000, 1], [90000, 4], [110000, 6],
    [800, 0], [2500, 0], [60000, 3], [75000, 5],
    [4000, 1], [10000, 0], [85000, 4], [130000, 6]
]

# 0 = Legitimate, 1 = Fraud
y = [
    0, 0, 0, 0,
    1, 1, 1, 1,
    0, 0, 1, 1,
    0, 0, 1, 1,
    0, 0, 1, 1
]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# Predict on unseen test data
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))