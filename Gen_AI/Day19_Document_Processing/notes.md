# Day 19 — Document Processing & Chunking

## 1. Document Ingestion Pipeline

Document ingestion converts raw documents into structured information that can be searched and retrieved by a RAG application.

**Pipeline:**

1. Load documents.
2. Extract text.
3. Clean extracted text.
4. Split text into chunks.
5. Attach metadata.
6. Generate embeddings.
7. Store embeddings and associated data in a vector database.

## 2. Document Extraction

Different document formats require different extraction techniques.

| Document | Common tool |
|---|---|
| PDF | PyMuPDF (`fitz`) |
| DOCX | python-docx |
| CSV | pandas |
| HTML | BeautifulSoup |
| Scanned PDF | OCR such as Tesseract |

**Important:** A scanned PDF may contain images instead of machine-readable text. Ordinary PDF extraction may fail, so OCR is required.

## 3. Text Cleaning

Cleaning improves the quality of extracted text.

Common operations:
- Remove unnecessary whitespace.
- Remove repeated headers and footers.
- Remove unwanted page numbers.
- Fix broken line breaks.
- Normalise encoding.
- Preserve meaningful punctuation, dates, numbers and policy references.

Example:

```python
import re

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

Do not blindly remove numbers or punctuation because they may contain important policy conditions.

## 4. Document Chunking

Chunking means splitting a large document into smaller pieces before generating embeddings.

### Why chunk?

- Improves retrieval granularity.
- Avoids embedding unnecessarily large passages.
- Helps retrieve relevant information.
- Reduces unrelated context passed to the LLM.

### Chunk Size

Chunk size determines how much text each chunk contains.

A smaller chunk may provide focused information but lose surrounding context. A larger chunk may preserve context but introduce irrelevant information.

There is no universally optimal chunk size. It depends on the document, embedding model, retrieval strategy and use case.

### Chunk Overlap

Overlap repeats some content between consecutive chunks.

Example:
- Chunk size: 300 words.
- Overlap: 50 words.
- Next chunk starts 250 words after the previous starting position.

Benefits:
- Preserves context across chunk boundaries.
- Reduces the chance of splitting related information.

Excessive overlap increases storage, indexing costs and duplicate retrieval.

## 5. Chunking Strategies

### Fixed-Size Chunking
Splits text using a fixed number of words or tokens.

Advantages:
- Simple.
- Predictable.
- Easy to implement.

Disadvantages:
- May split sentences or paragraphs.

### Recursive Chunking
Attempts to split text using natural boundaries, generally in this order:
1. Paragraphs.
2. Sentences.
3. Smaller separators.
4. Individual words when necessary.

It is a practical starting strategy for many document-processing pipelines.

### Semantic Chunking
Groups text according to meaning or topic changes.

It can preserve semantic relationships but is more complex and may require additional computation.

## 6. Metadata

Metadata stores information associated with each chunk.

Example:

```python
{
    "id": "banking_doc_001",
    "text": "Late loan payments may attract penalty charges.",
    "metadata": {
        "document_name": "loan_policy.pdf",
        "page_number": 12,
        "category": "banking_policy",
        "chunk_number": 3,
        "version": "1.0"
    }
}
```

Benefits:
- Document and page citations.
- Source traceability.
- Filtering by category or document.
- Version management.
- Better retrieval organisation.

**Security note:** Metadata filtering is not a substitute for access control. The application must independently enforce document permissions.

## 7. Banking Document Processing

For banking policy documents, the ingestion pipeline should preserve:
- Policy numbers and versions.
- Effective dates.
- Page numbers.
- Headings and section relationships.
- Tables and their associated conditions.
- Department or document classification.

Scanned documents require OCR. Complex tables may require layout-aware extraction to preserve row and column relationships.

## 8. Practical Chunking Implementation

```python
def create_chunks(text, chunk_size=100, overlap=20):
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
```

The step size is:

`step = chunk_size - overlap`

The break prevents an unnecessary final chunk containing only repeated overlap words.

## Key Takeaways

- Document ingestion prepares raw documents for retrieval.
- Cleaning improves extracted text quality.
- Chunking improves retrieval granularity.
- Overlap helps preserve context across chunk boundaries.
- Recursive chunking respects natural text boundaries.
- Metadata supports citations, filtering and traceability.
- OCR is necessary when PDFs contain scanned images rather than extractable text.
