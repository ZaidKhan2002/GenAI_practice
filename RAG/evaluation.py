import math
import sys

from explore_dataset import build_pipeline
from retrieval import retrieve_bm25, retrieve_chunks, reciprocal_rank_fusion
from reranking import reranker, rerank_results

evaluation_dataset = [
    {
        "question": "What was JPMorgan's net income in 2024?",
        "expected_answer": "$58,471 million",
        "expected_page": 174,
        "expected_metric": "Net income",
        "expected_value": "58,471"
    },
    {
        "question": "What was JPMorgan's total net revenue in 2024?",
        "expected_answer": "$177,556 million",
        "expected_page": 174,
        "expected_metric": "Total net revenue",
        "expected_value": "177,556"
    },
    {
        "question": "What was JPMorgan's total noninterest expense in 2024?",
        "expected_answer": "$91,797 million",
        "expected_page": 174,
        "expected_metric": "Total noninterest expense",
        "expected_value": "91,797"
    },
    {
        "question": "What was JPMorgan's income tax expense in 2024?",
        "expected_answer": "$16,610 million",
        "expected_page": 174,
        "expected_metric": "Income tax expense",
        "expected_value": "16,610"
    },
    {
        "question": "What was JPMorgan's net income applicable to common stockholders in 2024?",
        "expected_answer": "$56,868 million",
        "expected_page": 174,
        "expected_metric": "Net income applicable to common stockholders",
        "expected_value": "56,868"
    },
    {
        "question": "How much did JPMorgan's net income increase from 2023 to 2024?",
        "expected_answer": "$8,919 million",
        "expected_page": 174,
        "expected_metric": "Net income",
        "expected_value": "58,471"
    },
    {
        "question": "What was the percentage growth in net income from 2023 to 2024?",
        "expected_answer": "Approximately 18.00%",
        "expected_page": 174,
        "expected_metric": "Net income",
        "expected_value": "58,471"
    },
    {
        "question": "How much did JPMorgan's total net revenue increase from 2023 to 2024?",
        "expected_answer": "$19,452 million",
        "expected_page": 174,
        "expected_metric": "Total net revenue",
        "expected_value": "177,556"
    },
    {
        "question": "How much did JPMorgan's total noninterest expense increase from 2022 to 2024?",
        "expected_answer": "$15,657 million",
        "expected_page": 174,
        "expected_metric": "Total noninterest expense",
        "expected_value": "91,797"
    },
    {
        "question": "How much did JPMorgan's net interest income increase from 2023 to 2024?",
        "expected_answer": "$3,316 million",
        "expected_page": 174,
        "expected_metric": "Net interest income",
        "expected_value": "92,583"
    },
    {
        "question": "What was JPMorgan's interest income in 2024?",
        "expected_answer": "$193,933 million",
        "expected_page": 174,
        "expected_metric": "Interest income",
        "expected_value": "193,933"
    },
    {
        "question": "What was JPMorgan's interest expense in 2024?",
        "expected_answer": "$101,350 million",
        "expected_page": 174,
        "expected_metric": "Interest expense",
        "expected_value": "101,350"
    },
    {
        "question": "What was JPMorgan's net interest income in 2024?",
        "expected_answer": "$92,583 million",
        "expected_page": 174,
        "expected_metric": "Net interest income",
        "expected_value": "92,583"
    },
    {
        "question": "What was JPMorgan's provision for credit losses in 2024?",
        "expected_answer": "$10,678 million",
        "expected_page": 174,
        "expected_metric": "Provision for credit losses",
        "expected_value": "10,678"
    },
    {
        "question": "What was JPMorgan's income before income tax expense in 2024?",
        "expected_answer": "$75,081 million",
        "expected_page": 174,
        "expected_metric": "Income before income tax expense",
        "expected_value": "75,081"
    },
    {
        "question": "What was the difference between JPMorgan's interest income and interest expense in 2024?",
        "expected_answer": "$92,583 million",
        "expected_page": 174,
        "expected_evidence": [
            {"metric": "Interest income", "value": "193,933"},
            {"metric": "Interest expense", "value": "101,350"}
        ]
    },
    {
        "question": "What was the difference between JPMorgan's net income and net income applicable to common stockholders in 2024?",
        "expected_answer": "$1,603 million",
        "expected_page": 174,
        "expected_evidence": [
            {"metric": "Net income", "value": "58,471"},
            {"metric": "Net income applicable to common stockholders", "value": "56,868"}
        ]
    },
    {
        "question": "What was the difference between total net revenue and total noninterest expense in 2024?",
        "expected_answer": "$85,759 million",
        "expected_page": 174,
        "expected_evidence": [
            {"metric": "Total net revenue", "value": "177,556"},
            {"metric": "Total noninterest expense", "value": "91,797"}
        ]
    },
    {
        "question": "What was the combined amount of net interest income and noninterest revenue in 2024?",
        "expected_answer": "$177,556 million",
        "expected_page": 174,
        "expected_evidence": [
            {"metric": "Net interest income", "value": "92,583"},
            {"metric": "Noninterest revenue", "value": "84,973"}
        ]
    }
]

unsupported_questions = [
    {"question": "What was JPMorgan's net income in 2030?", "expected_behavior": "abstain"},
    {"question": "What was JPMorgan's net income in 2025?", "expected_behavior": "abstain"},
    {"question": "What was JPMorgan's revenue in 2010?", "expected_behavior": "abstain"},
    {"question": "How many employees did JPMorgan hire in 2024?", "expected_behavior": "abstain"},
    {"question": "What was JPMorgan's market share in the Indian banking industry in 2024?", "expected_behavior": "abstain"}
]


# ==========================================================
# GROUND-TRUTH HELPERS
# ==========================================================

def get_expected_evidence(evaluation_item):

    evidence = evaluation_item.get("expected_evidence")

    if evidence is None:
        evidence = [{
            "metric": evaluation_item["expected_metric"],
            "value": evaluation_item["expected_value"]
        }]

    return evidence


def is_relevant_chunk(chunk, evaluation_item):

    if chunk["document_id"] != "jpmorgan_chase_2024":
        return False

    if chunk["page"] != evaluation_item["expected_page"]:
        return False

    evidence = get_expected_evidence(evaluation_item)

    return any(
        item["metric"].lower() in chunk["chunk_text"].lower()
        and item["value"] in chunk["chunk_text"]
        for item in evidence
    )


def has_complete_evidence(retrieved_chunks, evaluation_item):

    evidence = get_expected_evidence(evaluation_item)

    retrieved_text = [chunk["chunk_text"].lower() for chunk in retrieved_chunks]

    return all(
        any(
            item["metric"].lower() in text and item["value"] in text
            for text in retrieved_text
        )
        for item in evidence
    )


# ==========================================================
# RETRIEVAL PIPELINE
# ==========================================================

def retrieve_all_methods(query, pipe):

    chunks = pipe["chunks"]

    bm25_results = retrieve_bm25(
        query, pipe["bm25_index"], chunks, top_k=20
    )

    semantic_results = retrieve_chunks(
        query, pipe["vector_index"], chunks, top_k=20
    )

    hybrid_candidates = reciprocal_rank_fusion(
        bm25_results, semantic_results, chunks, top_k=20
    )

    reranked_results = rerank_results(
        query, hybrid_candidates, reranker, top_k=5
    )

    return {
        "BM25": bm25_results[:5],
        "Semantic": semantic_results[:5],
        "Hybrid RRF": hybrid_candidates[:5],
        "Hybrid + Reranker": reranked_results
    }


# ==========================================================
# EVALUATION METRICS
# ==========================================================

def calculate_recall_at_k(retrieved_chunks, relevant_chunk_ids, k=5):

    if k <= 0:
        raise ValueError("k must be positive")

    if not relevant_chunk_ids:
        return 0.0

    retrieved_ids = {chunk["chunk_id"] for chunk in retrieved_chunks[:k]}

    return len(retrieved_ids & relevant_chunk_ids) / len(relevant_chunk_ids)


def calculate_reciprocal_rank(retrieved_chunks, relevant_chunk_ids):

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk["chunk_id"] in relevant_chunk_ids:
            return 1 / rank

    return 0.0


def calculate_precision_at_k(retrieved_chunks, relevant_chunk_ids, k=5):

    if k <= 0:
        raise ValueError("k must be positive")

    retrieved_ids = {chunk["chunk_id"] for chunk in retrieved_chunks[:k]}

    return len(retrieved_ids & relevant_chunk_ids) / k


def calculate_ndcg_at_k(retrieved_chunks, relevant_chunk_ids, k=5):

    if k <= 0:
        raise ValueError("k must be positive")

    if not relevant_chunk_ids:
        return 0.0

    relevance = [
        1 if chunk["chunk_id"] in relevant_chunk_ids else 0
        for chunk in retrieved_chunks[:k]
    ]

    dcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(relevance, start=1)
    )

    ideal_relevance = [1] * min(len(relevant_chunk_ids), k)

    idcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(ideal_relevance, start=1)
    )

    return dcg / idcg if idcg > 0 else 0.0


# ==========================================================
# EVALUATION RUNNER
# ==========================================================

def evaluate_retrieval(dataset, pipe):

    corpus = pipe["chunks"]

    methods = ["BM25", "Semantic", "Hybrid RRF", "Hybrid + Reranker"]

    evaluation_scores = {
        method: {
            "recall": [],
            "mrr": [],
            "precision": [],
            "ndcg": [],
            "complete_evidence": []
        }
        for method in methods
    }

    for question_number, item in enumerate(dataset, start=1):

        query = item["question"]

        relevant_chunk_ids = {
            chunk["chunk_id"]
            for chunk in corpus
            if is_relevant_chunk(chunk, item)
        }

        if not relevant_chunk_ids:
            print(f"WARNING Q{question_number}: no ground-truth chunks found "
                  f"for: {query}")

        retrieval_methods = retrieve_all_methods(query, pipe)

        for method, results in retrieval_methods.items():

            evaluation_scores[method]["recall"].append(
                calculate_recall_at_k(results, relevant_chunk_ids, k=5)
            )
            evaluation_scores[method]["mrr"].append(
                calculate_reciprocal_rank(results, relevant_chunk_ids)
            )
            evaluation_scores[method]["precision"].append(
                calculate_precision_at_k(results, relevant_chunk_ids, k=5)
            )
            evaluation_scores[method]["ndcg"].append(
                calculate_ndcg_at_k(results, relevant_chunk_ids, k=5)
            )
            evaluation_scores[method]["complete_evidence"].append(
                has_complete_evidence(results, item)
            )

    return evaluation_scores


def print_evaluation_summary(evaluation_scores):

    print("\n" + "=" * 75)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 75)

    for method, scores in evaluation_scores.items():

        if not scores["recall"]:
            continue

        total = len(scores["recall"])
        complete_count = sum(scores["complete_evidence"])

        print(f"\nMethod: {method}")
        print(f"Questions evaluated: {total}")
        print(f"Average Recall@5:    {sum(scores['recall']) / total:.4f}")
        print(f"Average MRR:         {sum(scores['mrr']) / total:.4f}")
        print(f"Average Precision@5: {sum(scores['precision']) / total:.4f}")
        print(f"Average NDCG@5:      {sum(scores['ndcg']) / total:.4f}")
        print(f"Complete Evidence:   {complete_count}/{total} "
              f"({complete_count / total:.2%})")


# ==========================================================
# OPTIONAL: ABSTENTION TEST (needs Ollama running)
# ==========================================================

def evaluate_abstention(questions, pipe):

    from validation import answer_rag_query

    correct = 0

    for item in questions:

        query = item["question"]

        results = retrieve_all_methods(query, pipe)["Hybrid + Reranker"]
        response = answer_rag_query(query, results)

        abstained = response.get("insufficient_evidence") is True
        correct += abstained

        print(f"[{'OK ' if abstained else 'BAD'}] {query}")
        print(f"      -> {response.get('answer')}")

    print(f"\nAbstained correctly: {correct}/{len(questions)}")


# ==========================================================
# EXECUTION
# ==========================================================

if __name__ == "__main__":

    pipe = build_pipeline()

    scores = evaluate_retrieval(evaluation_dataset, pipe)
    print_evaluation_summary(scores)

    if "--abstain" in sys.argv:
        print("\n" + "=" * 75)
        print("ABSTENTION TEST")
        print("=" * 75)
        evaluate_abstention(unsupported_questions, pipe)