import re
from explore_dataset import documents
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""

    lines = text.splitlines()
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
    text = '\n'.join(line for line in lines if line)

    text = re.sub(
        r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',
        '',
        text
    )

    return text.strip()


# Prepare cleaned documents
processed_documents = []

for document in documents:
    processed_documents.append({
        "document_id": document["document_id"],
        "page": document["page"],
        "raw_text": document["text"],
        "cleaned_text": clean_text(document["text"])
    })


# Create chunks from cleaned text
def create_chunks(text, chunk_size=300, overlap=50):
    if not isinstance(text, str) or not text.strip():
        return []

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size or overlap")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start += chunk_size - overlap

    return chunks

# Process documents to create chunks with metadata
def process_documents(documents):
    processed_chunks = []

    for document in documents:
        cleaned_text = document.get("cleaned_text", "")
        chunks = create_chunks(cleaned_text)

        for idx, chunk in enumerate(chunks):
            processed_chunks.append({
                "document_id": document["document_id"],
                "page": document["page"],
                "chunk_id": f"{document['document_id']}_p{document['page']}_c{idx}",
                "chunk_text": chunk
            })

    return processed_chunks

processed_chunks = process_documents(processed_documents)

# print("Total chunks:", len(processed_chunks))
# print("First chunk keys:", processed_chunks[0].keys())

# Generate embeddings for the chunks
def generate_embeddings(documents):
    if not documents:
        return []

    chunk_texts = [doc["chunk_text"] for doc in documents]

    embeddings = sentence_model.encode(
        chunk_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    if len(embeddings) != len(documents):
        raise ValueError("Embedding count mismatch")

    for doc, embedding in zip(documents, embeddings):
        doc["embedding"] = embedding

    return documents

embedded_chunks = generate_embeddings(processed_chunks)

# print("Total embedded chunks:", len(embedded_chunks))
# print("Embedding shape:", embedded_chunks[0]["embedding"].shape)

def create_vector_index(documents):
    if not documents:
        raise ValueError("No documents provided for indexing")

    embedding_dim = len(documents[0]["embedding"])

    embeddings = np.array(
        [doc["embedding"] for doc in documents],
        dtype="float32"
    )

    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    return index

index = create_vector_index(embedded_chunks)

# print("Total vectors:", index.ntotal)
# print("Embedding dimensions:", index.d)


def retrieve_chunks(query, index, documents, top_k=5):

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    if not documents:
        raise ValueError("No documents available")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    query_embedding = sentence_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0 or idx >= len(documents):
            continue

        doc = documents[idx]

        results.append({
            "score": float(score),
            "document_id": doc["document_id"],
            "page": doc["page"],
            "chunk_id": doc["chunk_id"],
            "chunk_text": doc["chunk_text"]
        })

    return results



# queries = [
#     "JPMorgan Chase net income for fiscal year 2024",
#     "Net income attributable to JPMorgan Chase and Co. in 2024",
#     "2024 annual consolidated statement of income net income"
# ]

# for query in queries:
#     print(f"\nQUERY: {query}")
#     results = retrieve_chunks(query, index, embedded_chunks, top_k=3)

#     for result in results:
#         print(
#             f"Score: {result['score']:.4f} | "
#             f"Page: {result['page']}"
#         )
#         print(result["chunk_text"][:250])

for doc in embedded_chunks:
    if doc["page"] == 174:
        print(doc["chunk_id"])
        print(doc["chunk_text"])
        print("-" * 80)