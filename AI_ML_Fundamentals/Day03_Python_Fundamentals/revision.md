# Day 3 — Quick Revision

## Data Structures

- List → Mutable.
- Tuple → Immutable.
- Dictionary → Key-value pairs.
- Set → Unique values.

## Functions

- `def` → Define function.
- `return` → Return result.
- `print()` → Display output.
- `f""` → Formatted string.

## Important Python Concepts

- `if / elif / else` → Conditions.
- `for / while` → Loops.
- `try / except` → Exception handling.
- `import` → Load modules.
- `venv` → Isolate dependencies.

## JSON

- `dumps()` → Python object to JSON string.
- `loads()` → JSON string to Python object.

## Environment Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install package_name
python -m pip freeze > requirements.txt
```

## Practical Completed

Transaction Analyzer:
- Stored transactions as dictionaries inside a list.
- Classified transactions using conditions.
- Calculated total using `sum()`.

Result: ₹409,200 across eight transactions.