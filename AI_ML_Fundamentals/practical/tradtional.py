def detect_fraud(amount, failed_logins):
    if amount > 100000 and failed_logins >= 3:
        return "Fraud"

    return "Legitimate"


print(detect_fraud(120000, 4))
print(detect_fraud(5000, 0))