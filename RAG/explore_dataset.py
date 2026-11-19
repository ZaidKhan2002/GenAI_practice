from collections import Counter

from datasets import load_dataset

from ingestion import (
    process_documents,
    generate_embeddings,
    create_vector_index,
)
from retrieval import create_bm25_index

TARGET_DOC = "jpmorgan_chase_2024"


def load_documents(target_doc=TARGET_DOC):
    """Load the pages of one document from the ViDoRe finance corpus."""

    dataset = load_dataset(
        "vidore/vidore_v3_finance_en",
        "corpus",
        split="test"
    )

    # Drop heavy columns (e.g. page images) so iterating is fast
    dataset = dataset.select_columns(
        ["doc_id", "page_number_in_doc", "markdown"]
    )

    pages = sorted(
        (row for row in dataset if row["doc_id"] == target_doc),
        key=lambda row: row["page_number_in_doc"]
    )

    return [
        {
            "document_id": page["doc_id"],
            "page": page["page_number_in_doc"],
            "text": page["markdown"],
        }
        for page in pages
    ]


_PIPELINE = None


def build_pipeline(force=False):
    """Load -> chunk -> embed -> FAISS + BM25. Built once per process."""

    global _PIPELINE

    if _PIPELINE is not None and not force:
        return _PIPELINE

    documents = load_documents()
    print(f"Loaded {len(documents)} pages")

    chunks = process_documents(documents)
    print("Chunk counts:", dict(Counter(c["chunk_type"] for c in chunks)))

    chunks = generate_embeddings(chunks)

    _PIPELINE = {
        "documents": documents,
        "chunks": chunks,
        "vector_index": create_vector_index(chunks),
        "bm25_index": create_bm25_index(chunks),
    }

    return _PIPELINE


if __name__ == "__main__":
    # Diagnostic only: checks data loading and table chunking (no embeddings).
    docs = load_documents()
    print(f"Pages loaded: {len(docs)}")

    chunks = process_documents(docs)
    print("Chunk counts:", dict(Counter(c["chunk_type"] for c in chunks)))

    page_174 = [c for c in chunks if c["page"] == 174]
    print(f"Chunks on page 174: {len(page_174)}")

    tables = [c for c in page_174 if c["chunk_type"] == "table"]
    if tables:
        print("\nSample table chunk from page 174:\n")
        print(tables[0]["chunk_text"])
    else:
        print("\nNO table chunks on page 174 -> table chunking is failing.")
        raw = next((d["text"] for d in docs if d["page"] == 174), "")
        print("\nRaw markdown of page 174 (first 1500 chars):\n")
        print(raw[:1500])