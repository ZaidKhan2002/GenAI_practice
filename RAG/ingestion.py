import re
import numpy as np
import faiss
import ollama

from collections import Counter
from explore_dataset import documents
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# --------------------------------------------------
# INITIALISE EMBEDDING MODEL
# --------------------------------------------------

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# --------------------------------------------------
# STEP 1: TEXT CLEANING
# --------------------------------------------------

def clean_text(text):

    # Handle empty or invalid input
    if not isinstance(text, str) or not text.strip():
        return ""

    # Split text into individual lines
    lines = text.splitlines()

    # Normalise multiple spaces and tabs
    lines = [
        re.sub(r'[ \t]+', ' ', line).strip()
        for line in lines
    ]

    # Remove empty lines while preserving line breaks
    text = '\n'.join(line for line in lines if line)

    # Remove unwanted control characters
    text = re.sub(
        r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',
        '',
        text
    )

    return text.strip()

# --------------------------------------------------
# STEP 2: BASIC TEXT CHUNKING
# --------------------------------------------------

def create_chunks(text, chunk_size=300, overlap=50):

    if not isinstance(text, str) or not text.strip():
        return []

    # Validate chunk size and overlap
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size or overlap")

    words = text.split()

    chunks = []
    start = 0

    while start < len(words):

        # Determine the end position
        end = min(start + chunk_size, len(words))

        # Create chunk
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)

        # Stop after processing all words
        if end == len(words):
            break

        # Move forward while retaining overlap
        start += chunk_size - overlap

    return chunks

# --------------------------------------------------
# STEP 3: EXTRACT MARKDOWN TABLE ROWS
# --------------------------------------------------

def extract_table_rows(markdown):

    rows = []

    for line in markdown.splitlines():

        line = line.strip()

        # Process only Markdown table rows
        if not line.startswith("|"):
            continue

        cells = [
            cell.strip()
            for cell in line.strip("|").split("|")
        ]

        # Preserve empty cells to maintain column alignment
        if any(cells):
            rows.append(cells)

    return rows

# --------------------------------------------------
# STEP 4: REMOVE TABLE ROWS FROM NARRATIVE TEXT
# --------------------------------------------------

def remove_table_lines(markdown):

    narrative_lines = []

    for line in markdown.splitlines():

        # Retain lines outside Markdown tables
        if not line.strip().startswith("|"):
            narrative_lines.append(line)

    return '\n'.join(narrative_lines).strip()

# --------------------------------------------------
# STEP 5: CREATE STRUCTURED TABLE CHUNKS
# --------------------------------------------------

def create_table_chunks(table_rows, document_id, page):

    # Require at least a title, reporting information,
    # headers and one data row
    if len(table_rows) < 4:
        return []

    # Validate the expected table structure
    if not all(
        len(row) == len(table_rows[2])
        for row in table_rows[:3]
    ):
        return []

    table_title = table_rows[1][0]
    reporting_info = table_rows[2]

    # Extract year headers
    headers = reporting_info[1:]

    # Accept only recognised four-digit year headers
    if not headers or not all(
        re.fullmatch(r'\d{4}', year)
        for year in headers
    ):
        return []

    chunks = []

    for row in table_rows[3:]:

        # Skip rows with unexpected column counts
        if len(row) != len(headers) + 1:
            continue

        metric = row[0]
        values = row[1:]

        # Skip empty or section-heading rows
        if not metric or not any(values):
            continue

        # Build a self-describing financial chunk
        row_text = (
            f"Document: {document_id}\n"
            f"Page: {page}\n"
            f"Statement: {table_title}\n"
            f"Reporting period: {reporting_info[0]}\n"
            f"Metric: {metric}\n"
        )

        # Preserve the correct year-value relationships
        for year, value in zip(headers, values):

            if value:
                row_text += f"{year}: {value}\n"

        chunks.append(row_text)

    return chunks

# --------------------------------------------------
# STEP 6: INTEGRATED HYBRID CHUNKING PIPELINE
# --------------------------------------------------

def process_documents(documents):

    processed_chunks = []

    for document in documents:

        raw_text = document["text"]
        document_id = document["document_id"]
        page = document["page"]

        # 1. Extract table rows
        table_rows = extract_table_rows(raw_text)

        # 2. Generate structured table chunks
        table_chunks = create_table_chunks(
            table_rows,
            document_id,
            page
        )

        # 3. Separate narrative content
        narrative_text = remove_table_lines(raw_text)

        # 4. Clean narrative content
        cleaned_text = clean_text(narrative_text)

        # 5. Generate ordinary narrative chunks
        narrative_chunks = create_chunks(cleaned_text)

        # 6. Store narrative chunks with metadata
        for idx, chunk in enumerate(narrative_chunks):

            processed_chunks.append({
                "document_id": document_id,
                "page": page,
                "chunk_id": f"{document_id}_p{page}_n{idx}",
                "chunk_type": "narrative",
                "chunk_text": chunk
            })

        # 7. Store table chunks with metadata
        for idx, chunk in enumerate(table_chunks):

            processed_chunks.append({
                "document_id": document_id,
                "page": page,
                "chunk_id": f"{document_id}_p{page}_t{idx}",
                "chunk_type": "table",
                "chunk_text": chunk
            })

    return processed_chunks

# --------------------------------------------------
# STEP 7: GENERATE EMBEDDINGS
# --------------------------------------------------

def generate_embeddings(documents):

    if not documents:
        return []

    # Extract chunk text
    chunk_texts = [
        doc["chunk_text"]
        for doc in documents
    ]

    # Generate normalised embeddings
    embeddings = sentence_model.encode(
        chunk_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Validate embedding count
    if len(embeddings) != len(documents):
        raise ValueError("Embedding count mismatch")

    # Attach embeddings to corresponding chunks
    for doc, embedding in zip(documents, embeddings):
        doc["embedding"] = embedding

    return documents

# --------------------------------------------------
# STEP 8: CREATE FAISS VECTOR INDEX
# --------------------------------------------------

def create_vector_index(documents):

    if not documents:
        raise ValueError("No documents provided for indexing")

    embedding_dim = len(documents[0]["embedding"])

    # Convert embeddings into float32 NumPy array
    embeddings = np.array(
        [doc["embedding"] for doc in documents],
        dtype="float32"
    )

    # Inner product on normalised vectors = cosine similarity
    index = faiss.IndexFlatIP(embedding_dim)

    index.add(embeddings)

    return index

# --------------------------------------------------
# STEP 9: SEMANTIC RETRIEVAL
# --------------------------------------------------

def retrieve_chunks(query, index, documents, top_k=5):

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    if not documents:
        raise ValueError("No documents available")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    # Convert query into an embedding
    query_embedding = sentence_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Retrieve the nearest chunks
    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        # Ignore invalid FAISS indices
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

# --------------------------------------------------
# STEP 10: EXECUTE HYBRID CHUNKING
# --------------------------------------------------

hybrid_chunks = process_documents(documents)

# Uncomment to inspect chunk generation

# print("Total chunks:", len(hybrid_chunks))

# print("Chunk distribution:")
# print(Counter(
#     chunk["chunk_type"] for chunk in hybrid_chunks
# ))

# print("First chunk:")
# print(hybrid_chunks[0]["chunk_text"])
# for chunk in hybrid_chunks:
#     if (
#         chunk["page"] == 174
#         and chunk["chunk_type"] == "table"
#     ):
#         print(chunk["chunk_text"])
#         print("-" * 50)

# --------------------------------------------------
# STEP 11: GENERATE EMBEDDINGS
# --------------------------------------------------

embedded_chunks = generate_embeddings(hybrid_chunks)

# Uncomment to inspect embedding generation

# print("Total embedded chunks:", len(embedded_chunks))
# print("Embedding shape:",
#       embedded_chunks[0]["embedding"].shape)

# --------------------------------------------------
# STEP 12: CREATE FAISS INDEX
# --------------------------------------------------

index = create_vector_index(embedded_chunks)

# Uncomment to inspect FAISS index

# print("Total vectors:", index.ntotal)
# print("Embedding dimensions:", index.d)

# --------------------------------------------------
# STEP 13: TEST SEMANTIC RETRIEVAL
# --------------------------------------------------

# queries = [
#     "JPMorgan Chase net income for fiscal year 2024",
#     "Net income attributable to JPMorgan Chase and Co. in 2024",
#     "2024 annual consolidated statement of income net income"
# ]

# for query in queries:

#     print(f"\nQUERY: {query}")

#     results = retrieve_chunks(
#         query,
#         index,
#         embedded_chunks,
#         top_k=3
#     )

#     for result in results:

#         print(
#             f"Score: {result['score']:.4f} | "
#             f"Page: {result['page']} | "
#             f"Type: {result['chunk_type']}"
#         )

#         print(result["chunk_text"][:250])


# --------------------------------------------------
# STEP 14: BM25 RETRIEVAL
# --------------------------------------------------

def tokenize_text(text):
    """
    Convert text into lowercase word and number tokens.
    Keep comma-separated financial numbers together.
    """
    return re.findall(
        r"\d[\d,]*(?:\.\d+)?|[a-zA-Z]+",
        text.lower()
    )


def create_bm25_index(documents):
    if not documents:
        raise ValueError("No documents provided for BM25 indexing")

    # Tokenize every chunk
    tokenized_corpus = [
        tokenize_text(doc["chunk_text"])
        for doc in documents
    ]

    # Build the BM25 index
    bm25_index = BM25Okapi(tokenized_corpus)

    return bm25_index

# Create BM25 keyword search index
bm25_index = create_bm25_index(hybrid_chunks)

def retrieve_bm25(query, bm25_index, documents, top_k=5):

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    if not documents:
        raise ValueError("No documents available")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    # Tokenize the user query
    tokenized_query = tokenize_text(query)

    # Calculate BM25 scores
    scores = bm25_index.get_scores(tokenized_query)

    # Get indices of highest-scoring chunks
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

# query = "Net income JPMorgan Chase consolidated statements of income 2024"

# bm25_results = retrieve_bm25(
#     query,
#     bm25_index,
#     hybrid_chunks,
#     top_k=5
# )

# for result in bm25_results:
#     print(
#         f"Page: {result['page']} | "
#         f"Score: {result['score']:.4f} | "
#         f"Type: {result['chunk_type']}"
#     )
#     print(result["chunk_text"][:400])
#     print("-" * 60)

# --------------------------------------------------
# STEP 15: RECIPROCAL RANK FUSION (RRF)
# --------------------------------------------------

def reciprocal_rank_fusion(
    bm25_results,
    semantic_results,
    documents,
    k=60,
    top_k=5
):
    fused_scores = {}

    if k <= 0 or top_k <= 0:
        raise ValueError("k and top_k must be positive")
        
    # Process BM25 ranking
    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["chunk_id"]

        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

    # Process semantic ranking
    for rank, result in enumerate(semantic_results, start=1):
        chunk_id = result["chunk_id"]

        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

    # Map chunk IDs to their original documents
    document_lookup = {
        doc["chunk_id"]: doc
        for doc in documents
    }

    ranked_chunks = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    results = []

    for chunk_id, score in ranked_chunks[:top_k]:
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

# query = "Net income JPMorgan Chase consolidated statements of income 2024"

# bm25_results = retrieve_bm25(
#     query, bm25_index, hybrid_chunks, top_k=10
# )

# semantic_results = retrieve_chunks(
#     query, index, embedded_chunks, top_k=10
# )

# hybrid_results = reciprocal_rank_fusion(
#     bm25_results,
#     semantic_results,
#     hybrid_chunks,
#     top_k=5
# )

# for result in hybrid_results:
#     print("RRF Score:", round(result["rrf_score"], 5))
#     print("Page:", result["page"])
#     print("Type:", result["chunk_type"])
#     print("Chunk:", result["chunk_text"])
#     print("-" * 60)

# --------------------------------------------------
# STEP 16: RERANKING WITH CROSS-ENCODER
# --------------------------------------------------

def rerank_results(query, candidates, reranker, top_k=5):

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    if not candidates:
        return []

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    # Prepare query-document pairs
    pairs = [
        (query, result["chunk_text"])
        for result in candidates
    ]

    # Calculate cross-encoder relevance scores
    scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=True
    )

    # Attach scores to candidate results
    reranked_results = []

    for result, score in zip(candidates, scores):
        reranked_result = result.copy()
        reranked_result["cross_encoder_score"] = float(score)

        reranked_results.append(reranked_result)

    # Sort by relevance score (highest first)
    reranked_results.sort(
        key=lambda result: result["cross_encoder_score"],
        reverse=True
    )

    return reranked_results[:top_k]

# Display final results
# for rank, result in enumerate(reranked_results, start=1):
#     print(f"Rank: {rank}")
#     print("Cross-encoder score:", round(result["cross_encoder_score"], 4))
#     print("Page:", result["page"])
#     print("Type:", result["chunk_type"])
#     print("Chunk:", result["chunk_text"])
#     print("-" * 60)

# --------------------------------------------------
# STEP 17: RAG PROMPT CREATION
# --------------------------------------------------

def create_rag_prompt(query, retrieved_chunks):

    context = ""

    for rank, result in enumerate(retrieved_chunks, start=1):
        context += f"""
Source {rank}:
Document: {result['document_id']}
Page: {result['page']}
Chunk type: {result['chunk_type']}

Content:
{result['chunk_text']}

"""

    prompt = f"""
You are a financial document analysis assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Identify the financial metric specifically requested.
2. Prefer the primary consolidated financial statement
   over supplementary statements when both are available.
3. Do not mix figures from different financial statements.
4. Do not include related financial metrics unless requested.
5. Preserve financial figures and reporting periods accurately.
6. Cite the exact page supporting your answer.
7. Keep the answer concise.
8. If sufficient evidence is unavailable, say:
   "The provided documents do not contain sufficient information to answer this question."

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt

def has_year_evidence(query, retrieved_chunks):

    match = re.search(r"\b(?:19|20)\d{2}\b", query)

    if not match:
        return True

    requested_year = match.group()

    for result in retrieved_chunks:
        if requested_year in result["chunk_text"]:
            return True

    return False

def answer_rag_query(query, retrieved_chunks):

    if not retrieved_chunks:
        return "No relevant information was retrieved."

    if not has_year_evidence(query, retrieved_chunks):
        return (
            "The retrieved document context does not contain "
            "financial information for the requested year."
        )

    prompt = create_rag_prompt(query, retrieved_chunks)

    return generate_answer(prompt)

def generate_answer(prompt):

    response = ollama.chat(
        model="qwen3.5:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    return response["message"]["content"]

query = "What was JPMorgan's net income in 2024?"

# Step 1: BM25 retrieval
bm25_results = retrieve_bm25(
    query, bm25_index, hybrid_chunks, top_k=20
)

# Step 2: FAISS semantic retrieval
semantic_results = retrieve_chunks(
    query, index, embedded_chunks, top_k=20
)

# Step 3: Reciprocal Rank Fusion
hybrid_results = reciprocal_rank_fusion(
    bm25_results,
    semantic_results,
    hybrid_chunks,
    top_k=20
)

# Step 4: Cross-encoder reranking
reranked_results = rerank_results(
    query,
    hybrid_results,
    reranker,
    top_k=5
)

# Step 5: RAG generation with evidence validation
answer = answer_rag_query(
    query,
    reranked_results
)

print("RAG Answer:")
print(answer)