from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Customers must repay loans according to the repayment schedule.",
    "Borrowers may request loan restructuring under specified conditions.",
    "Savings account interest is calculated quarterly.",
    "Late loan payments may attract penalty charges."
]

query = "What happens if a borrower misses a loan payment?"

# TODO 1: Generate document embeddings
document_embeddings = model.encode(documents, normalize_embeddings=True)

# TODO 2: Generate query embedding
query_embedding = model.encode(query,normalize_embeddings=True)

# TODO 3: Calculate cosine similarity between query and every document
cosine_similarities = np.dot(
    document_embeddings,
    query_embedding
)

# TODO 4: Rank documents by similarity
ranked_indices = np.argsort(cosine_similarities.flatten())[::-1]

# TODO 5: Retrieve Top-2 documents and print their content
top_k = 2
top_k_indices = ranked_indices[:top_k]
print("Top-2 most similar documents:")
for idx in top_k_indices:
    print(f"Document: {documents[idx]} | Similarity Score: {cosine_similarities[idx][0]:.4f}")