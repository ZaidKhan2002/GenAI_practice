# Day 3 — Interview Questions

1. **List vs Tuple?**  
   Lists are mutable; tuples are immutable.

2. **List vs Dictionary?**  
   A list stores ordered elements; a dictionary stores key-value pairs.

3. **What is a Set?**  
   A collection that stores unique elements.

4. **What is the difference between `return` and `print()`?**  
   `return` sends a value back from a function; `print()` displays it.

5. **What is a Python virtual environment?**  
   An isolated environment for managing project-specific dependencies.

6. **What is exception handling?**  
   Handling runtime errors using `try`, `except` and optionally `finally`.

7. **What is an f-string?**  
   A way to embed variables and expressions inside formatted strings.

8. **What is the difference between `json.dumps()` and `json.loads()`?**  
   `dumps()` converts a Python object to a JSON string; `loads()` parses a JSON string into a Python object.

9. **What is a Python module?**  
   A Python file containing reusable code that can be imported.

10. **Why is Python widely used in AI engineering?**  
    Its ecosystem provides libraries for data processing, ML, deep learning, LLM applications and API development.

11. **What is a generator expression?**  
    An expression that produces values lazily instead of creating an entire collection in memory.

    Example:
    ```python
    total = sum(t["amount"] for t in transactions)
    ```