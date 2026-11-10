import re
from explore_dataset import documents


def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""

    lines = text.splitlines()
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
    text = '\n'.join(line for line in lines if line)

    text = re.sub(
        r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',
        '',
        text
    )

    return text.strip()


# Prepare cleaned documents
processed_documents = []

for document in documents:
    processed_documents.append({
        "document_id": document["document_id"],
        "page": document["page"],
        "raw_text": document["text"],
        "cleaned_text": clean_text(document["text"])
    })


# Create chunks from cleaned text
def create_chunks(text, chunk_size=300, overlap=50):
    if not isinstance(text, str) or not text.strip():
        return []

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size or overlap")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start += chunk_size - overlap

    return chunks

# Process documents to create chunks with metadata
def process_documents(documents):
    processed_documents = []

    for document in documents:
        cleaned_text = document.get("cleaned_text", "")
        chunks = create_chunks(cleaned_text)

        for idx, chunk in enumerate(chunks):
            processed_documents.append({
                "document_id": document["document_id"],
                "page": document["page"],
                "chunk_id": f"{document['document_id']}_p{document['page']}_c{idx}",
                "chunk_text": chunk
            })

    return processed_documents

