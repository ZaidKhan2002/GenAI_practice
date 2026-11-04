import numpy as np

Q = np.array([
    [1, 0],
    [0, 1]
])

K = np.array([
    [1, 0],
    [0, 1],
    [1, 1]
])

V = np.array([
    [10, 0],
    [0, 10],
    [5, 5]
])

def softmax(x, axis=-1):
    # TODO: subtract maximum for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    
    # TODO: calculate exponentials
    x_exp = np.exp(x_shifted)
    
    # TODO: divide by sum along the given axis
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    return x_exp / x_sum


def scaled_dot_product_attention(Q, K, V):
    # TODO 1: Calculate Q multiplied by K transpose
    scores = np.dot(Q, K.T)

    # TODO 2: Divide scores by sqrt(d_k)
    d_k = K.shape[-1]
    scores = scores / np.sqrt(d_k)

    # TODO 3: Apply softmax row-wise
    attention_weights = softmax(scores, axis=-1)

    # TODO 4: Multiply attention weights by V
    output = np.dot(attention_weights, V)

    # TODO 5: Return output and attention weights
    return output, attention_weights

#  Call the function using the supplied Q, K and V matrices and print:
scores = np.dot(Q, K.T)
output, attention_weights = scaled_dot_product_attention(Q, K, V)

# 1. Raw scores
print("Raw scores:")
print(scores)

# 2. Scaled scores
print("Scaled scores:")
print(scores / np.sqrt(K.shape[-1]))

# 3. Attention weights
print("Attention weights:")
print(attention_weights)

# 4. Final output
print("Final output:")
print(output)

# 5. Sum of each attention-weight row
print("Sum of each attention-weight row:")
print(np.sum(attention_weights, axis=-1))