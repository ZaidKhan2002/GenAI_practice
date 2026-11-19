from sentence_transformers import CrossEncoder

# Model used to score query-document relevance
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_results(query, candidates, reranker, top_k=5):
    """Reorder candidate chunks based on Cross-Encoder relevance."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    if not candidates:
        return []

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    pairs = [(query, result["chunk_text"]) for result in candidates]

    scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False
    )

    reranked_results = []

    for result, score in zip(candidates, scores):
        reranked_result = result.copy()
        reranked_result["cross_encoder_score"] = float(score)
        reranked_results.append(reranked_result)

    reranked_results.sort(
        key=lambda result: result["cross_encoder_score"],
        reverse=True
    )

    return reranked_results[:top_k]