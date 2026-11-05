import numpy as np

tokens = ["machine", "rules", "analytics", "customer", "weather"]

logits = np.array([3.2, 2.5, 1.5, 0.8, 0.2])

# Softmax with temperature
def softmax(logits, temperature=1.0):
    # Divide logits by temperature.
    logits = logits / temperature
    # Apply numerically stable Softmax.
    exp_logits = np.exp(logits - np.max(logits))
    # Return probabilities.
    return exp_logits / np.sum(exp_logits)

def greedy_decode(tokens, probabilities):
    # Get the index of the token with the highest probability.
    max_index = np.argmax(probabilities)
    # Return the token corresponding to that index.
    return tokens[max_index]


def top_k_sampling(tokens, probabilities, k):
    if k < 1 or k > len(tokens):
        raise ValueError("k must be between 1 and the number of tokens")

    # Get indices of the k largest probabilities
    indices = np.argpartition(probabilities, -k)[-k:]

    # Sort selected indices by descending probability
    indices = indices[np.argsort(probabilities[indices])[::-1]]

    # Copy and normalise probabilities
    top_k_probabilities = probabilities[indices].copy()
    top_k_probabilities /= np.sum(top_k_probabilities)

    # Python lists cannot be indexed by NumPy arrays
    selected_tokens = [tokens[i] for i in indices]

    return selected_tokens, top_k_probabilities


def top_p_sampling(tokens, probabilities, p):
    if not 0 < p <= 1:
        raise ValueError("p must be greater than 0 and at most 1")

    # Sort indices by descending probability
    sorted_indices = np.argsort(probabilities)[::-1]

    # Calculate cumulative probabilities
    cumulative = np.cumsum(probabilities[sorted_indices])

    # Find the first index reaching or exceeding p
    cutoff = np.searchsorted(cumulative, p, side="left")

    # Include the token that crosses the threshold
    top_p_indices = sorted_indices[:cutoff + 1]

    # Copy and normalise retained probabilities
    top_p_probabilities = probabilities[top_p_indices].copy()
    top_p_probabilities /= np.sum(top_p_probabilities)

    selected_tokens = [tokens[i] for i in top_p_indices]

    return selected_tokens, top_p_probabilities


probabilities = softmax(logits, temperature=1.0)

print("Probabilities:", probabilities)
print("Greedy:", greedy_decode(tokens, probabilities))

print("Top-K:", top_k_sampling(tokens, probabilities, 3))
print("Top-P:", top_p_sampling(tokens, probabilities, 0.80))