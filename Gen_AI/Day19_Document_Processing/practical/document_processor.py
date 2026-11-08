document = """
LOAN POLICY

Customers must repay loans according to
the repayment schedule.

Late loan payments may attract penalty charges.

Borrowers may request loan restructuring
under specified conditions.

SAVINGS POLICY

Savings account interest is calculated quarterly.
"""

import re

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

cleaned_document = clean_text(document)

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

created_chunks = create_chunks(cleaned_document, chunk_size=20, overlap=5)
print("Created Chunks:")
for idx, chunk in enumerate(created_chunks):
    print(f"Chunk {idx + 1}: {chunk}")