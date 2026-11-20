from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from datasets import load_dataset
from ingestion import process_documents, generate_embeddings, create_vector_index
from retrieval import create_bm25_index, retrieve_chunks, retrieve_bm25, reciprocal_rank_fusion
from reranking import rerank_results
from validation import answer_rag_query
from sentence_transformers import CrossEncoder


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application starting up...")

    # Load dataset
    dataset = load_dataset(
        "vidore/vidore_v3_finance_en",
        "corpus",
        split="test"
    )

    # Select JPMorgan 2024 annual report
    target_doc = "jpmorgan_chase_2024"

    pages = sorted(
        [
            row for row in dataset
            if row["doc_id"] == target_doc
        ],
        key=lambda row: row["page_number_in_doc"]
    )

    # Convert dataset rows into ingestion format
    documents = [
        {
            "document_id": page["doc_id"],
            "page": page["page_number_in_doc"],
            "text": page["markdown"]
        }
        for page in pages
    ]

    # Process documents into narrative and table chunks
    processed_chunks = process_documents(documents)
    embedded_chunks = generate_embeddings(processed_chunks)

    # Store processed chunks in application state
    app.state.documents = embedded_chunks
    app.state.vector_index = create_vector_index(app.state.documents)
    app.state.bm25_index = create_bm25_index(app.state.documents)
    app.state.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print(f"Loaded {len(documents)} pages")
    print(f"Created {len(processed_chunks)} chunks")

    yield

    print("Application shutting down...")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=500
    )


class QuestionResponse(BaseModel):
    question: str
    message: str


@app.post("/ask")
def ask_question(request: QuestionRequest):

    query = request.question

    # 1. Semantic retrieval
    semantic_results = retrieve_chunks(
        query,
        app.state.vector_index,
        app.state.documents,
        top_k=5
    )

    # 2. BM25 retrieval
    bm25_results = retrieve_bm25(
        query,
        app.state.bm25_index,
        app.state.documents,
        top_k=5
    )

    # 3. Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(
        bm25_results,
        semantic_results,
        app.state.documents,
        k=60,
        top_k=5
    )

    # 4. Cross-Encoder reranking
    reranked_results = rerank_results(
        query,
        fused_results,
        app.state.reranker,
        top_k=5
    )

    # 5. Generation + validation
    response = answer_rag_query(
        query,
        reranked_results
    )

    return response