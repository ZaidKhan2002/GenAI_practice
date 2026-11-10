from datasets import load_dataset

dataset = load_dataset(
    "vidore/vidore_v3_finance_en",
    "corpus",
    split="test"
)

target_doc = "jpmorgan_chase_2024"

pages = sorted(
    [
        row for row in dataset
        if row["doc_id"] == target_doc
    ],
    key=lambda row: row["page_number_in_doc"]
)

documents = []

for page in pages:
    documents.append({
        "document_id": page["doc_id"],
        "page": page["page_number_in_doc"],
        "text": page["markdown"]
    })

