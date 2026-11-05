# Day 13 — Quick Revision

## Prompt Engineering
Designing instructions and context to guide LLM outputs without changing model weights.

## Prompt Anatomy
- Role: Who should the model act as?
- Task: What should it do?
- Context: What information does it need?
- Constraints: What boundaries must it follow?
- Examples: What output pattern should it follow?
- Format: How should the response be structured?

## Prompting Types

| Type | Examples supplied |
|---|---:|
| Zero-shot | 0 |
| One-shot | 1 |
| Few-shot | Multiple |

## Important Distinctions

- Prompt engineering ≠ fine-tuning.
- Delimiters ≠ prompt-injection protection.
- Valid JSON ≠ valid schema.
- Valid schema ≠ factual correctness.
- LLM recommendation ≠ authorised backend action.

## Structured Output Validation

1. Parse JSON using `json.loads()`.
2. Validate keys and data types.
3. Validate business rules.
4. Reject invalid responses.

## Prompt Injection Defences

- Treat retrieved content as untrusted.
- Separate instructions from data.
- Enforce backend access control.
- Restrict tool permissions.
- Validate outputs.
- Use human approval for consequential actions.

## Regression Testing

Test representative cases after prompt modifications:
- Normal inputs.
- Missing information.
- Contradictory evidence.
- Malicious input.
- Invalid output.
- Edge cases.

## Interview One-Liners

- Few-shot examples guide behaviour without updating model weights.
- JSON mode focuses on syntax; structured output constrains schema.
- Backend validation protects application correctness.
- Prompts are part of application behaviour and should be regression-tested.
- An LLM should not be the application's security boundary.
