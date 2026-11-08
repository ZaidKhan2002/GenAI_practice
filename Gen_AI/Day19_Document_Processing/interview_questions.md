# Day 19 — Interview Questions

## 1. Why do we split documents into chunks before generating embeddings?

Large documents are not ideal retrieval units. Chunking creates smaller, focused passages that can be embedded and retrieved independently. This improves retrieval granularity and helps avoid passing unrelated content to the LLM.

## 2. What is chunk overlap, and why is it important?

Chunk overlap repeats a portion of the previous chunk in the next chunk. It preserves context when related sentences or information are split across chunk boundaries.

Excessive overlap increases storage costs and duplicate retrieval.

## 3. What is the difference between fixed-size and recursive chunking?

Fixed-size chunking splits text based on a predefined number of words or tokens. It may split sentences or paragraphs.

Recursive chunking attempts to split using natural boundaries such as paragraphs and sentences before using smaller separators.

## 4. Why is metadata important in a RAG application?

Metadata provides additional information about each chunk, such as document ID, page number, category and version. It supports filtering, citations, traceability and document management.

## 5. How would you process a scanned banking policy PDF?

Ordinary PDF extraction may fail because scanned PDFs contain images instead of machine-readable text.

I would use OCR to extract text, apply layout-aware processing for tables, preserve page numbers and headings, clean the extracted content, and then perform chunking and embedding.

## 6. What happens if chunk size is too small?

Chunks may lose important surrounding context. A single policy condition might become separated from its explanation or exception.

## 7. What happens if chunk size is too large?

Retrieved chunks may contain irrelevant information, consume more context-window capacity and reduce retrieval precision.

## 8. Why should we preserve page numbers and document versions?

Page numbers enable source citations and traceability. Document versions help ensure the application retrieves information from the correct policy revision rather than outdated documents.

## 9. Does metadata filtering guarantee document security?

No. Metadata filtering helps narrow retrieval results, but the application must enforce authentication and authorisation independently. Access permissions should be checked before exposing retrieved content to the user or LLM.

## 10. Why is table extraction important for banking documents?

Tables may contain relationships between amounts, conditions, dates and eligibility criteria. Extracting values without preserving their row and column relationships can change the meaning of a policy.