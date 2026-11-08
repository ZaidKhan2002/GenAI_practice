from sentence_transformers import SentenceTransformer
import Gen_AI.Day18_VectorDB.practical.vector_search as vector_search
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Customers must repay loans according to the repayment schedule.",
    "Borrowers may request loan restructuring under specified conditions.",
    "Savings account interest is calculated quarterly.",
    "Late loan payments may attract penalty charges."
]

# Generate normalised embeddings
embeddings = model.encode(
    documents,
    normalize_embeddings=True
)

embeddings = np.array(embeddings, dtype="float32")

# Create a FAISS index
dimension = embeddings.shape[1]
index = vector_search.IndexFlatIP(dimension)

# Store embeddings in the index
index.add(embeddings)

# Embed the query
query = "What happens if a borrower misses a loan payment?"

query_embedding = model.encode(
    [query],
    normalize_embeddings=True
).astype("float32")

# Search Top-2
scores, indices = index.search(query_embedding, k=2)

for score, idx in zip(scores[0], indices[0]):
    print(f"Document: {documents[idx]}")
    print(f"Similarity: {score:.4f}")