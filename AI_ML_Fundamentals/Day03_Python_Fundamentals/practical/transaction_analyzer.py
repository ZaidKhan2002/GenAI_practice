transaction_data = [
    { "id": 1, "amount": 500, "failed_logins": 0 },
    { "id": 2, "amount": 1000, "failed_logins": 0 },
    { "id": 3, "amount": 95000, "failed_logins": 5 },
    { "id": 4, "amount": 80000, "failed_logins": 4 },
    { "id": 5, "amount": 700, "failed_logins": 1 },
    { "id": 6, "amount": 120000, "failed_logins": 0 },
    { "id": 7, "amount": 110000, "failed_logins": 5 },
    { "id": 8, "amount": 2000, "failed_logins": 0 }
]

def classify_transaction(transaction):
    if transaction["amount"] > 100000 and transaction["failed_logins"] >= 3:
        return "High Risk"

    elif transaction["amount"] > 50000:
        return "Medium Risk"

    else:
        return "Low Risk"

for transaction in transaction_data:
    classification = classify_transaction(transaction)
    print(f"Transaction ID: {transaction['id']}, Amount: {transaction['amount']}, Classification: {classification}")


def analyze_transactions(transactions):
    total_amount = sum(t["amount"] for t in transactions)
    print(f"Total Transaction Amount: {total_amount}")


analyze_transactions(transaction_data)
