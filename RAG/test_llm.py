import ollama

response = ollama.chat(
    model="qwen3.5:4b",
    messages=[
        {
            "role": "user",
            "content": "Explain what net income means in one sentence."
        }
    ]
)

print(response["message"]["content"])