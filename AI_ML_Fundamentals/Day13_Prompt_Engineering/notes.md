# Day 13 — Prompt Engineering

## 1. What is Prompt Engineering?

Prompt engineering is the process of designing and improving instructions given to an LLM to obtain useful, relevant and consistent outputs.

A prompt may contain:
- Role
- Task
- Context
- Constraints
- Examples
- Output format

Prompt engineering guides model behaviour at inference time. It does not update model weights.

## 2. Anatomy of a Prompt

| Component | Purpose |
|---|---|
| Role | Establishes the model's perspective |
| Task | Defines what the model should do |
| Context | Supplies relevant information |
| Constraints | Defines boundaries and restrictions |
| Examples | Demonstrates expected behaviour |
| Output format | Specifies the response structure |

### Instruction hierarchy

- System: High-level governing instructions.
- Developer: Application-level instructions.
- User: User's request and input.

A lower-priority instruction should not override a higher-priority instruction. Exact implementations vary across providers.

## 3. Prompting Techniques

### Zero-shot prompting
Asking the model to perform a task without providing examples.

### One-shot prompting
Providing one input-output example before the actual task.

### Few-shot prompting
Providing multiple examples to demonstrate the desired behaviour.

Examples guide the current inference. They do not fine-tune the model or update its weights.

## 4. Prompt Design Best Practices

- Use clear and specific instructions.
- Supply sufficient context.
- Separate instructions from input data.
- Define explicit constraints.
- Specify the expected output format.
- Handle missing or insufficient information.
- Avoid contradictory examples and instructions.

### Delimiters

Delimiters such as triple quotes, XML-like tags and code fences help separate input data from instructions.

They improve clarity but are not a security boundary.

## 5. Reasoning Prompts

Chain-of-Thought (CoT) prompting is a technique that encourages structured problem-solving.

For application development, a structured analysis template can request:

1. Observed symptoms.
2. Available evidence.
3. Possible causes.
4. Evidence supporting each cause.
5. Recommended actions.
6. Missing information.

A detailed explanation does not guarantee correctness. Prefer concise, verifiable rationales and explicit evidence.

## 6. Structured Outputs

LLM responses may be returned in machine-readable formats such as JSON.

### JSON mode vs structured output

| JSON mode | Structured output |
|---|---|
| Generally ensures JSON syntax | Constrains output to a defined schema |
| Does not necessarily enforce required keys | Can enforce required properties |
| Does not necessarily enforce types | Can constrain data types |
| Still requires validation | Still requires validation |

Structured output does not guarantee factual accuracy.

### Three validation layers

1. Syntax validation: Is the response valid JSON?
2. Schema validation: Are the required keys and data types correct?
3. Business validation: Does the response follow application-specific rules?

## 7. Prompt Injection

Prompt injection occurs when malicious content attempts to manipulate an LLM into violating its intended instructions.

Potential sources:
- User input.
- Retrieved RAG documents.
- Uploaded files.
- Web pages.
- Tool outputs.

Defence in depth:
- Treat external content as untrusted data.
- Use explicit trust boundaries in prompts.
- Enforce backend authentication and authorisation.
- Apply least-privilege tool access.
- Validate model outputs.
- Require human approval for consequential actions.

A prompt must never replace backend security controls.

## 8. Prompt Evaluation and Regression Testing

Prompts influence application behaviour and should be tested like other application components.

A practical evaluation workflow:

1. Build representative test cases.
2. Define expected behaviour.
3. Run the existing prompt.
4. Modify the prompt.
5. Run the same test cases.
6. Compare output quality, format compliance and security.
7. Investigate regressions.

LLM evaluation should consider correctness and behaviour rather than relying only on exact string matching.

## 9. Practical Implementation

File: `practical/structured_output.py`

Implemented a simulated LLM response-validation pipeline using Python's built-in JSON library.

Functions:
- `parse_response()`
- `validate_schema()`
- `validate_business_rules()`
- `process_llm_response()`

Key concepts:
- `json.loads()` converts JSON strings into Python objects.
- `json.dumps()` converts Python objects into JSON strings.
- Schema validation checks keys, types and allowed values.
- Business validation checks application-specific conditions.

## 10. Key Takeaways

- Better prompts reduce ambiguity.
- Examples guide model responses without updating weights.
- Delimiters improve clarity but cannot guarantee security.
- Structured output improves integration reliability.
- Backend validation remains essential.
- Prompt changes require regression testing.
- LLMs should assist operational decisions, not independently control critical systems.