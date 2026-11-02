from sklearn import linear_model as lr
import pandas as pd

data = {
    "training_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "productivity_score": [25, 30, 40, 45, 50, 60, 65, 75, 80, 90]
}

df = pd.DataFrame(data)

# Separate the input feature (X) and target (y).
X = df[["training_hours"]]
y = df["productivity_score"]

# Create a Linear Regression model
model = lr.LinearRegression()

# Train the model using .fit().
model.fit(X, y)

# Predict the score for 11 training hours.
new_hours = pd.DataFrame({
    "training_hours": [11]
})

predicted_score = model.predict(new_hours)

print(f"Predicted productivity score : {predicted_score[0]}, model coefficient : {model.coef_[0]}, model intercept : {model.intercept_}")