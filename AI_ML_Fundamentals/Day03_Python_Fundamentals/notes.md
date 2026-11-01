# Day 3 — Python Fundamentals

## 1. Python Basics

- Python is dynamically typed: variable types are determined at runtime.
- `print()` displays output.
- `type()` checks a variable's type.
- Python uses indentation to define code blocks.
- Python is strongly typed: incompatible types require explicit conversion.

## 2. Basic Data Types

| Type | Example |
|---|---|
| int | `25` |
| float | `99.5` |
| str | `"Hello"` |
| bool | `True` |
| None | `None` |

## 3. Python Data Structures

| Structure | Characteristics |
|---|---|
| List | Ordered, mutable collection |
| Tuple | Ordered, immutable collection |
| Dictionary | Key-value pairs |
| Set | Collection of unique elements |

Example: A transaction can be represented using a dictionary, and multiple transactions can be stored in a list.

## 4. Core Programming Concepts

- Conditions: `if`, `elif`, `else`
- Loops: `for`, `while`
- Functions: `def`
- Return: Sends a result back and exits the function.
- Print: Displays output.
- Exception handling: `try`, `except`, `finally`
- f-strings: Format strings using variables, e.g. `f"Amount: {amount}"`.

## 5. Modules and Environments

- `import`: Loads a module.
- `pip`: Installs external Python packages.
- `venv`: Creates an isolated project environment.
- `requirements.txt`: Records project dependencies.

Commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas
python -m pip freeze > requirements.txt
```

## 6. JSON Handling

Using Python's built-in `json` module:

- `json.dumps()` — Python object to JSON string.
- `json.loads()` — JSON string to Python object.

Useful for exchanging data with APIs and LLM applications.

## 7. Practical: Transaction Analyzer

Implemented in `practical/transaction_analyzer.py`.

Concepts used:
- List of dictionaries.
- Functions and conditional statements.
- Loops and f-strings.
- `sum()` with a generator expression.

**Output:** Correctly classified eight transactions and calculated a total of ₹409,200.