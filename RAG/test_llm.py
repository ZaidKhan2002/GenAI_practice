import sys
import ollama

from explore_dataset import build_pipeline
from generation import MODEL_NAME
from retrieval import (
    retrieve_chunks,
    retrieve_bm25,
    reciprocal_rank_fusion,
)
from reranking import reranker, rerank_results
from validation import answer_rag_query

def ping_llm():
    """Quick check that Ollama is running and the model responds."""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": "Explain what net income means in one sentence."
            }
        ]
    )

    print(response["message"]["content"])


def run_query(query, pipe, verbose=True):
    chunks = pipe["chunks"]

    # Semantic + keyword retrieval
    semantic_results = retrieve_chunks(
        query, pipe["vector_index"], chunks, top_k=20
    )
    bm25_results = retrieve_bm25(
        query, pipe["bm25_index"], chunks, top_k=20
    )

    # Hybrid fusion
    fused_chunks = reciprocal_rank_fusion(
        bm25_results, semantic_results, chunks, top_k=20
    )

    # Reranking (the reranker must be passed explicitly)
    reranked_chunks = rerank_results(query, fused_chunks, reranker, top_k=5)

    if verbose:
        print("\nTop chunks sent to the LLM:")
        for r in reranked_chunks:
            print(
                f"  {r['chunk_id']:<40} {r['chunk_type']:<9} "
                f"page={r['page']:<4} score={r['cross_encoder_score']:.3f}"
            )

    # Generate and validate
    return answer_rag_query(query, reranked_chunks)


if __name__ == "__main__":

    # python test_llm.py --ping   -> only test Ollama, skip the RAG pipeline
    if "--ping" in sys.argv:
        ping_llm()
        sys.exit(0)

    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What was JPMorgan's net income in 2024?"
    )

    pipe = build_pipeline()

    print(f"\nQuestion: {query}")
    response = run_query(query, pipe)

    print("\nRAG Response:")
    print(response)