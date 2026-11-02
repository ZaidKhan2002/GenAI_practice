import numpy as np

# Inputs and weights
x = np.array([0.8, 0.6])
w = np.array([0.5, 0.9])
b = -0.4

# Step 1: Calculate the weighted sum
weighted_sum = np.dot(x, w)

# Step 2: Add the bias
z = weighted_sum + b

# Step 3: Apply ReLU
final_output = np.maximum(0, z)

# Step 4: Print the final output
print("Final Output:", final_output)