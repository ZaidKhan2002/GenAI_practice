import re
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# STEP 1: TEXT CLEANING

def clean_text(text):
    """Clean extracted text while preserving its line structure."""

    if not isinstance(text, str) or not text.strip():
        return ""

    lines = text.splitlines()
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]

    text = "\n".join(line for line in lines if line)

    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    return text.strip()


# STEP 2: NARRATIVE CHUNKING

def create_chunks(text, chunk_size=300, overlap=50):
    """Split text into word-based chunks with overlapping context."""

    if not isinstance(text, str) or not text.strip():
        return []

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size or overlap")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start += chunk_size - overlap

    return chunks


# STEP 3: TABLE EXTRACTION

def extract_table_rows(markdown):
    """Extract Markdown table rows while preserving empty cells."""

    if not isinstance(markdown, str) or not markdown.strip():
        return []

    rows = []

    for line in markdown.splitlines():
        line = line.strip()

        if not line.startswith("|"):
            continue

        cells = [cell.strip() for cell in line.strip("|").split("|")]

        if any(cells):
            rows.append(cells)

    return rows


# STEP 4: REMOVE TABLES FROM NARRATIVE TEXT

def remove_table_lines(markdown):
    """Remove Markdown table lines, keeping narrative text."""

    if not isinstance(markdown, str) or not markdown.strip():
        return ""

    narrative_lines = []

    for line in markdown.splitlines():
        if not line.strip().startswith("|"):
            narrative_lines.append(line)

    return "\n".join(narrative_lines).strip()


# STEP 5: TABLE-AWARE CHUNKING

def create_table_chunks(table_rows, document_id, page):
    """Convert financial table rows into self-describing chunks."""

    if len(table_rows) < 4:
        return []

    reporting_info = table_rows[2]

    if not all(len(row) == len(reporting_info) for row in table_rows[3:]):
        return []

    table_title = table_rows[1][0]

    headers = reporting_info[1:]

    if not headers or not all(re.fullmatch(r"\d{4}", year) for year in headers):
        return []

    chunks = []

    for row in table_rows[3:]:

        if len(row) != len(headers) + 1:
            continue

        metric = row[0]
        values = row[1:]

        if not metric or not any(values):
            continue

        row_text = (
            f"Document: {document_id}\n"
            f"Page: {page}\n"
            f"Statement: {table_title}\n"
            f"Reporting period: {reporting_info[0]}\n"
            f"Metric: {metric}\n"
        )

        for year, value in zip(headers, values):
            if value:
                row_text += f"{year}: {value}\n"

        chunks.append(row_text)

    return chunks


# STEP 6: DOCUMENT PROCESSING

def process_documents(documents):
    """Create narrative and table chunks with metadata."""

    processed_chunks = []

    for document in documents:

        raw_text = document["text"]
        document_id = document["document_id"]
        page = document["page"]

        table_rows = extract_table_rows(raw_text)
        table_chunks = create_table_chunks(table_rows, document_id, page)

        narrative_text = remove_table_lines(raw_text)
        cleaned_text = clean_text(narrative_text)
        narrative_chunks = create_chunks(cleaned_text)

        for idx, chunk in enumerate(narrative_chunks):
            processed_chunks.append({
                "document_id": document_id,
                "page": page,
                "chunk_id": f"{document_id}_p{page}_n{idx}",
                "chunk_type": "narrative",
                "chunk_text": chunk
            })

        for idx, chunk in enumerate(table_chunks):
            processed_chunks.append({
                "document_id": document_id,
                "page": page,
                "chunk_id": f"{document_id}_p{page}_t{idx}",
                "chunk_type": "table",
                "chunk_text": chunk
            })

    return processed_chunks


# STEP 7: EMBEDDING GENERATION

_sentence_model = None


def get_sentence_model():
    """Return the shared embedding model, loading it lazily."""

    global _sentence_model

    if _sentence_model is None:
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    return _sentence_model


def generate_embeddings(documents):
    """Generate normalized embeddings and attach them to chunks."""

    if not documents:
        return []

    chunk_texts = [doc["chunk_text"] for doc in documents]

    embeddings = get_sentence_model().encode(
        chunk_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    if len(embeddings) != len(documents):
        raise ValueError("Embedding count mismatch")

    for doc, embedding in zip(documents, embeddings):
        doc["embedding"] = embedding

    return documents


# STEP 8: FAISS VECTOR INDEX

def create_vector_index(documents):
    """Build a FAISS index from normalized embeddings."""

    if not documents:
        raise ValueError("No documents provided for indexing")

    if any("embedding" not in doc for doc in documents):
        raise ValueError("Missing embeddings in documents")

    embedding_dim = len(documents[0]["embedding"])

    if any(len(doc["embedding"]) != embedding_dim for doc in documents):
        raise ValueError("Inconsistent embedding dimensions")

    embeddings = np.asarray(
        [doc["embedding"] for doc in documents],
        dtype="float32"
    )

    # Inner product of normalized vectors == cosine similarity
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    return index