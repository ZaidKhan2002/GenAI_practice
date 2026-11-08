import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Customer failed to repay the loan.",
    "Borrower missed the scheduled loan payment.",
    "The weather is pleasant today."
]

# TODO 1: Generate embeddings
sentences_embeddings = model.encode(sentences)

# TODO 2: Print embedding shape
print("Embedding shape:", sentences_embeddings.shape)

# TODO 3: Print the embedding of the first sentence
print("Embedding of the first sentence:", sentences_embeddings[0])

# TODO 4: Compare the embeddings of the first two sentences using cosine similarity
A = sentences_embeddings[0]
B = sentences_embeddings[2]

cosine_similarity = np.dot(A, B) / (
    np.linalg.norm(A) * np.linalg.norm(B)
)

print("Cosine similarity:", cosine_similarity)