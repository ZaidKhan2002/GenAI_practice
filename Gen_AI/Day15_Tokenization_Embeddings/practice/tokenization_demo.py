from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "The customer failed to repay the loan."

# 1. Convert text into tokens
tokens = tokenizer.tokenize(text)

# 2. Generate complete tokenizer output
complete_output = tokenizer(text, return_tensors="pt")

# Extract token IDs
token_ids = complete_output["input_ids"]

# 3. Convert token IDs back into tokens
reconstructed_tokens = tokenizer.convert_ids_to_tokens(
    token_ids[0].tolist()
)

# 4. Print results
print("Tokens:", tokens)
print("Number of tokens:", len(tokens))
print("Token IDs:", token_ids)
print("Reconstructed tokens:", reconstructed_tokens)
print("Complete output:", complete_output)

# 5. Print attention mask
print("Attention Mask:", complete_output["attention_mask"])


uncommon_words = [
    "microfinance",
    "unbankability",
    "electroencephalographically"
]

for word in uncommon_words:
    print(word, "->", tokenizer.tokenize(word))