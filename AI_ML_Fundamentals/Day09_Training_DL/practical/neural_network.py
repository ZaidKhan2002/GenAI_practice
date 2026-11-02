import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

class FraudDetectionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# Create the model
model = FraudDetectionModel()

# Training features: amount, failed_logins
X = [
    [500, 0],
    [1000, 0],
    [2000, 1],
    [1500, 0],
    [95000, 5],
    [80000, 4],
    [70000, 3],
    [120000, 5],
    [3000, 0],
    [5000, 1],
    [90000, 4],
    [110000, 6]
]

# Labels: 0 = Legitimate, 1 = Fraud
y = [
    0, 0, 0, 0,
    1, 1, 1, 1,
    0, 0, 1, 1
]

# Standardise features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert data into PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)


# Create dataset and batches
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Create model
model = FraudDetectionModel()

# Binary classification loss
criterion = nn.BCELoss()

# Adam optimiser
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100

for epoch in range(epochs):

    for batch_X, batch_y in loader:

        optimizer.zero_grad()

        predictions = model(batch_X)

        loss = criterion(predictions, batch_y)

        loss.backward()

        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Switch model to evaluation mode
model.eval()

with torch.no_grad():
    predictions = model(X_tensor)

    predicted_classes = (predictions >= 0.5).float()

    print("\nPredicted probabilities:")
    print(predictions)

    print("\nPredicted classes:")
    print(predicted_classes)

    print("\nActual labels:")
    print(y_tensor)


