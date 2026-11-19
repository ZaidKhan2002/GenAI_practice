import re
import numpy as np

from rank_bm25 import BM25Okapi
from ingestion import get_sentence_model

# STEP 1: SEMANTIC RETRIEVAL

def retrieve_chunks(query, index, documents, top_k=5):
    """Retrieve relevant chunks using FAISS semantic similarity."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    if not documents:
        raise ValueError("No documents available")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    query_embedding = get_sentence_model().encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(query_embedding, dtype="float32")

    if query_embedding.shape[1] != index.d:
        raise ValueError("Query embedding dimension mismatch")

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
            "chunk_type": doc["chunk_type"],
            "chunk_text": doc["chunk_text"]
        })

    return results


# STEP 2: TEXT TOKENIZATION

def tokenize_text(text):
    """Lowercase word/number tokens; keeps comma-separated and decimal numbers."""

    if not isinstance(text, str):
        return []

    return re.findall(
        r"\d[\d,]*(?:\.\d+)?|[a-zA-Z]+",
        text.lower()
    )


# STEP 3: BM25 INDEX CREATION

def create_bm25_index(documents):
    """Create a BM25 index from document chunks."""

    if not documents:
        raise ValueError("No documents provided for BM25 indexing")

    tokenized_corpus = [
        tokenize_text(doc["chunk_text"])
        for doc in documents
    ]

    return BM25Okapi(tokenized_corpus)


# STEP 4: BM25 KEYWORD RETRIEVAL

def retrieve_bm25(query, bm25_index, documents, top_k=5):
    """Retrieve relevant chunks using BM25 keyword relevance."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    if not documents:
        raise ValueError("No documents available")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    tokenized_query = tokenize_text(query)

    if not tokenized_query:
        return []

    scores = bm25_index.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:

        doc = documents[idx]

        results.append({
            "score": float(scores[idx]),
            "document_id": doc["document_id"],
            "page": doc["page"],
            "chunk_id": doc["chunk_id"],
            "chunk_type": doc["chunk_type"],
            "chunk_text": doc["chunk_text"]
        })

    return results


# STEP 5: RECIPROCAL RANK FUSION (RRF)

def reciprocal_rank_fusion(
    bm25_results,
    semantic_results,
    documents,
    k=60,
    top_k=5
):
    """Combine BM25 and semantic rankings using RRF."""

    if k <= 0 or top_k <= 0:
        raise ValueError("k and top_k must be positive")

    fused_scores = {}

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["chunk_id"]
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1 / (k + rank)

    for rank, result in enumerate(semantic_results, start=1):
        chunk_id = result["chunk_id"]
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1 / (k + rank)

    document_lookup = {doc["chunk_id"]: doc for doc in documents}

    ranked_chunks = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    results = []

    for chunk_id, score in ranked_chunks[:top_k]:

        if chunk_id not in document_lookup:
            continue

        doc = document_lookup[chunk_id]

        results.append({
            "rrf_score": score,
            "document_id": doc["document_id"],
            "page": doc["page"],
            "chunk_id": chunk_id,
            "chunk_type": doc["chunk_type"],
            "chunk_text": doc["chunk_text"]
        })

    return results