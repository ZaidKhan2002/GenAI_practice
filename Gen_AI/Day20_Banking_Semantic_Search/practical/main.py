documents = [
    {
        "id": "POL001",
        "title": "Loan Repayment Policy",
        "page": 1,
        "text": """
        Customers must repay loans according to the repayment schedule.
        Late loan payments may attract penalty charges.
        Borrowers may request loan restructuring under specified conditions.
        """
    },
    {
        "id": "POL002",
        "title": "Savings Account Policy",
        "page": 2,
        "text": """
        Savings account interest is calculated quarterly.
        Customers must maintain the required minimum balance.
        Interest rates may change according to bank policy.
        """
    },
    {
        "id": "POL003",
        "title": "KYC Policy",
        "page": 3,
        "text": """
        Customers must complete identity verification before account activation.
        Aadhaar or other accepted identity documents may be used for verification.
        Periodic KYC updates may be required.
        """
    },
    {
        "id": "POL004",
        "title": "Digital Banking Policy",
        "page": 4,
        "text": """
        Customers must never share their OTP or banking passwords.
        Suspicious digital transactions should be reported immediately.
        The bank may temporarily restrict transactions during security investigations.
        """
    }
]

import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip().lower()

def create_chunks(text, chunk_size=30, overlap=5):
    words = text.split()
    chunks = []

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size or overlap")

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

        if i + chunk_size >= len(words):
            break

    return chunks

processed_chunks = []
for doc in documents:
    cleaned_text = clean_text(doc['text'])
    chunks = create_chunks(cleaned_text)
    for index, chunk in enumerate(chunks):
        processed_chunks.append({
            "text": chunk,
            "metadata": {
                "document_id": doc["id"],
                "title": doc["title"],
                "page": doc["page"],
                "chunk_number": index + 1
            }
        })

# print("Cleaned text:")
# print(repr(clean_text(documents[0]["text"])))

# print("\nWords:", clean_text(documents[0]["text"]).split())

# print("\nFirst document chunks:")
# for chunk in create_chunks(clean_text(documents[0]["text"])):
#     print(chunk)


model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in processed_chunks]

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

# print("Embedding shape:", embeddings.shape)

# Convert embeddings to FAISS-compatible format
embeddings = np.asarray(embeddings, dtype="float32")

# Get embedding dimension
dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatIP(dimension)

# Add document embeddings
index.add(embeddings)

# print("Total vectors stored:", index.ntotal)
# print("Index dimension:", index.d)


def search_documents(query, top_k=2):
    # Convert query into an embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Search FAISS index
    scores, indices = index.search(query_embedding, top_k)

    # Display retrieved documents
    for score, idx in zip(scores[0], indices[0]):
        result = processed_chunks[idx]

        print("\nSimilarity Score:", round(float(score), 4))
        print("Document:", result["metadata"]["title"])
        print("Page:", result["metadata"]["page"])
        print("Chunk:", result["metadata"]["chunk_number"])
        print("Content:", result["text"])

query = "How can customers protect their digital banking credentials?"

search_documents(query, top_k=2)